"""
High-performance ranking model with dual-head internal RRF,
temporal half-life alignment, and 76-dimensional feature space.
"""

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import Config
from src.rank.ranker import Ranker


class EmbeddingRanker(Ranker):
    """
    Enhanced ranking engine incorporating multimodal embeddings and temporal decay alignment.
    """

    def __init__(self, train_data, articles, customers, mappings,
                 item_similarity_dict=None, image_emb=None, text_emb=None):
        super().__init__(train_data, articles, customers, mappings, item_similarity_dict)
        self.image_emb = image_emb or {}
        self.text_emb = text_emb or {}
        print(f"EmbeddingRanker initialized: image_emb={len(self.image_emb):,}, text_emb={len(self.text_emb):,}")

        self.models_ranker = []
        self.models_classifier = []

        self.article_to_product_code = mappings.get('article_to_product_code', {})
        self.article_to_category = mappings.get('article_to_category', {})
        self.article_to_dept = articles.set_index('article_id')['department_name'].to_dict() if 'department_name' in articles.columns else {}
        self.article_to_group = articles.set_index('article_id')['product_group_name'].to_dict() if 'product_group_name' in articles.columns else {}

        user_age_dict = customers.set_index('customer_id')['age'].fillna(30).to_dict()

        def get_age_bin(age):
            if age < 25: return 0
            elif age < 35: return 1
            elif age < 45: return 2
            elif age < 55: return 3
            else: return 4

        self.user_age_bin = {uid: get_age_bin(age) for uid, age in user_age_dict.items()}

        self.user_avg_price = train_data.groupby('customer_id')['price'].mean().to_dict()
        self.user_std_price = train_data.groupby('customer_id')['price'].std().fillna(0.01).to_dict()
        self.article_price = train_data.groupby('article_id')['price'].mean().to_dict()

        train_data_cat = train_data[['customer_id', 'article_id', 'price']].copy()
        train_data_cat['cat'] = train_data_cat['article_id'].map(self.article_to_category)
        cat_price_stats = train_data_cat.groupby('cat')['price'].agg(['mean', 'std']).fillna(0)
        self.cat_price_mean = cat_price_stats['mean'].to_dict()
        self.cat_price_std = cat_price_stats['std'].replace(0, 0.01).to_dict()

        user_total = train_data.groupby('customer_id').size()
        user_cat = train_data_cat.groupby(['customer_id', 'cat']).size()
        self.user_cat_affinity = (user_cat / user_cat.index.get_level_values('customer_id').map(user_total)).to_dict()

        if self.article_to_dept:
            train_data_cat['dept'] = train_data_cat['article_id'].map(self.article_to_dept)
            user_dept = train_data_cat.groupby(['customer_id', 'dept']).size()
            self.user_dept_affinity = (user_dept / user_dept.index.get_level_values('customer_id').map(user_total)).to_dict()
        else:
            self.user_dept_affinity = {}

        if self.article_to_group:
            train_data_cat['group'] = train_data_cat['article_id'].map(self.article_to_group)
            user_group = train_data_cat.groupby(['customer_id', 'group']).size()
            self.user_group_affinity = (user_group / user_group.index.get_level_values('customer_id').map(user_total)).to_dict()
        else:
            self.user_group_affinity = {}

        print("Precomputing temporal context (as of 2020-09-15)...")
        self.default_context = self._create_temporal_context(self.train_data)
        self._build_user_embeddings()

    def _create_temporal_context(self, hist_data):
        max_date = hist_data['t_dat'].max()

        tmp_time = hist_data[['customer_id', 'article_id', 't_dat']].copy()
        tmp_time['days_ago'] = (max_date - tmp_time['t_dat']).dt.days
        tmp_time['decay_weight'] = (0.5) ** (tmp_time['days_ago'] / 7.0)
        decay_dict = tmp_time.groupby(['customer_id', 'article_id'])['decay_weight'].sum().to_dict()

        user_purchased = hist_data.groupby('customer_id')['article_id'].apply(set).to_dict()

        tmp_pc = hist_data[['customer_id', 'article_id']].copy()
        tmp_pc['pc'] = tmp_pc['article_id'].map(self.article_to_product_code)
        user_pcs = tmp_pc.groupby('customer_id')['pc'].apply(set).to_dict()

        win_1d = hist_data[hist_data['t_dat'] >= max_date - pd.Timedelta(days=1)]
        win_3d = hist_data[hist_data['t_dat'] >= max_date - pd.Timedelta(days=3)]
        win_7d = hist_data[hist_data['t_dat'] >= max_date - pd.Timedelta(days=7)]
        win_14d = hist_data[hist_data['t_dat'] >= max_date - pd.Timedelta(days=14)]
        win_30d = hist_data[hist_data['t_dat'] >= max_date - pd.Timedelta(days=30)]

        s1d = win_1d['article_id'].value_counts().to_dict()
        s3d = win_3d['article_id'].value_counts().to_dict()
        s7d = win_7d['article_id'].value_counts().to_dict()
        s14d = win_14d['article_id'].value_counts().to_dict()
        s30d = win_30d['article_id'].value_counts().to_dict()

        last_sale = hist_data.groupby('article_id')['t_dat'].max()
        last_sale_days = ((max_date - last_sale).dt.days).to_dict()

        last_7d_age = win_7d[['customer_id', 'article_id']].copy()
        last_7d_age['age_bin'] = last_7d_age['customer_id'].map(self.user_age_bin).fillna(1).astype(int)
        age_sales_7d = last_7d_age.groupby(['age_bin', 'article_id']).size().to_dict()

        return {
            'decay_repurchase_score': decay_dict,
            'user_purchased_hist': user_purchased,
            'user_product_codes_hist': user_pcs,
            'item_sales_1d': s1d,
            'item_sales_3d': s3d,
            'item_sales_7d': s7d,
            'item_sales_14d': s14d,
            'item_sales_30d': s30d,
            'item_last_sale_days': last_sale_days,
            'age_item_sales_7d': age_sales_7d,
        }

    def _build_user_embeddings(self):
        self.user_image_emb = {}
        self.user_text_emb = {}
        user_hist = self.default_context['user_purchased_hist']
        for uid, items in tqdm(user_hist.items(), desc="Computing user embeddings", leave=False):
            items_list = list(items)[-10:]
            image_vecs = [self.image_emb[a] for a in items_list if a in self.image_emb]
            if image_vecs:
                v = np.mean(image_vecs, axis=0)
                n = np.linalg.norm(v)
                self.user_image_emb[uid] = v / n if n > 0 else v

            text_vecs = [self.text_emb[a] for a in items_list if a in self.text_emb]
            if text_vecs:
                v = np.mean(text_vecs, axis=0)
                n = np.linalg.norm(v)
                self.user_text_emb[uid] = v / n if n > 0 else v

    def build_features(self, df, context=None):
        df = super().build_features(df)
        ctx = context if context is not None else self.default_context

        cids = df['customer_id'].values
        aids = df['article_id'].values

        user_hist = ctx['user_purchased_hist']
        user_pcs = ctx['user_product_codes_hist']
        art_pc = self.article_to_product_code

        df['decay_repurchase_score'] = np.array([
            ctx['decay_repurchase_score'].get((uid, aid), 0.0) for uid, aid in zip(cids, aids)
        ], dtype=np.float32)

        df['is_repurchased'] = np.array([
            1 if aid in user_hist.get(uid, set()) else 0 for uid, aid in zip(cids, aids)
        ], dtype=np.int8)

        df['is_product_code_bought'] = np.array([
            1 if art_pc.get(aid) in user_pcs.get(uid, set()) else 0 for uid, aid in zip(cids, aids)
        ], dtype=np.int8)

        u_prices = np.array([self.user_avg_price.get(uid, 0.028) for uid in cids], dtype=np.float32)
        u_stds = np.array([self.user_std_price.get(uid, 0.01) for uid in cids], dtype=np.float32)
        i_prices = np.array([self.article_price.get(aid, 0.028) for aid in aids], dtype=np.float32)

        df['price_diff'] = i_prices - u_prices
        df['price_ratio'] = i_prices / (u_prices + 1e-5)
        df['price_user_zscore'] = np.abs(i_prices - u_prices) / (u_stds + 1e-4)

        cats = [self.article_to_category.get(aid) for aid in aids]
        c_means = np.array([self.cat_price_mean.get(c, 0.028) for c in cats], dtype=np.float32)
        c_stds = np.array([self.cat_price_std.get(c, 0.01) for c in cats], dtype=np.float32)
        df['price_cat_zscore'] = (i_prices - c_means) / (c_stds + 1e-5)

        cat_aff = np.array([self.user_cat_affinity.get((uid, c), 0.0) for uid, c in zip(cids, cats)], dtype=np.float32)
        df['user_cat_affinity'] = cat_aff

        if self.article_to_dept:
            depts = [self.article_to_dept.get(aid) for aid in aids]
            dept_aff = np.array([self.user_dept_affinity.get((uid, d), 0.0) for uid, d in zip(cids, depts)], dtype=np.float32)
            df['user_dept_affinity'] = dept_aff
        else:
            dept_aff = np.zeros(len(df), dtype=np.float32)

        if self.article_to_group:
            groups = [self.article_to_group.get(aid) for aid in aids]
            group_aff = np.array([self.user_group_affinity.get((uid, g), 0.0) for uid, g in zip(cids, groups)], dtype=np.float32)
            df['user_group_affinity'] = group_aff
        else:
            group_aff = np.zeros(len(df), dtype=np.float32)

        df['novel_alignment_score'] = (cat_aff * 0.4 + dept_aff * 0.4 + group_aff * 0.2).astype(np.float32)

        age_bins = [self.user_age_bin.get(uid, 1) for uid in cids]
        df['age_cohort_trend_7d'] = np.array([
            ctx['age_item_sales_7d'].get((abin, aid), 0) for abin, aid in zip(age_bins, aids)
        ], dtype=np.float32)

        df['item_days_since_last_sale'] = np.array([
            ctx['item_last_sale_days'].get(aid, 999) for aid in aids
        ], dtype=np.float32)
        df['item_active_3d'] = (df['item_days_since_last_sale'] <= 3).astype(np.int8)
        df['item_active_7d'] = (df['item_days_since_last_sale'] <= 7).astype(np.int8)

        s1d = np.array([ctx['item_sales_1d'].get(aid, 0) for aid in aids], dtype=np.float32)
        s3d = np.array([ctx['item_sales_3d'].get(aid, 0) for aid in aids], dtype=np.float32)
        s7d = np.array([ctx['item_sales_7d'].get(aid, 0) for aid in aids], dtype=np.float32)
        s14d = np.array([ctx['item_sales_14d'].get(aid, 0) for aid in aids], dtype=np.float32)
        s30d = np.array([ctx['item_sales_30d'].get(aid, 0) for aid in aids], dtype=np.float32)

        df['item_sales_1d'] = s1d
        df['item_sales_3d'] = s3d
        df['item_sales_7d'] = s7d
        df['item_sales_14d'] = s14d
        df['item_burst_3d_7d'] = (s3d * 7.0) / (s7d * 3.0 + 1e-5)
        df['item_burst_7d_30d'] = (s7d * 30.0) / (s30d * 7.0 + 1e-5)

        if 'candidate_rank' in df.columns:
            df['candidate_rank_discount'] = (1.0 / np.log2(df['candidate_rank'] + 2.0)).astype(np.float32)

        image_sims, text_sims, image_norms, text_norms = [], [], [], []
        for uid, aid in zip(cids, aids):
            if uid in self.user_image_emb and aid in self.image_emb:
                u_vec, a_vec = self.user_image_emb[uid], self.image_emb[aid]
                image_sims.append(float(np.dot(u_vec, a_vec)))
                image_norms.append(float(np.linalg.norm(a_vec)))
            else:
                image_sims.append(0.0)
                image_norms.append(0.0)

            if uid in self.user_text_emb and aid in self.text_emb:
                u_vec, a_vec = self.user_text_emb[uid], self.text_emb[aid]
                text_sims.append(float(np.dot(u_vec, a_vec)))
                text_norms.append(float(np.linalg.norm(a_vec)))
            else:
                text_sims.append(0.0)
                text_norms.append(0.0)

        df['user_image_emb_sim'] = np.array(image_sims, dtype=np.float32)
        df['user_text_emb_sim'] = np.array(text_sims, dtype=np.float32)
        df['image_emb_norm'] = np.array(image_norms, dtype=np.float32)
        df['text_emb_norm'] = np.array(text_norms, dtype=np.float32)
        df['emb_sim_sum'] = (df['user_image_emb_sim'] + df['user_text_emb_sim']).astype(np.float32)

        return df

    def _sample_for_week(self, val_start, val_end, candidates, train_users, max_per_user=100):
        val_data = self.train_data[
            (self.train_data['t_dat'] >= val_start) &
            (self.train_data['t_dat'] < val_end)
        ]
        val_purchases = val_data.groupby('customer_id')['article_id'].apply(set).to_dict()

        hist_before_val = self.train_data[self.train_data['t_dat'] < val_start]
        temporal_ctx = self._create_temporal_context(hist_before_val)

        user_hist_ctx = temporal_ctx['user_purchased_hist']
        user_pcs_ctx = temporal_ctx['user_product_codes_hist']

        rows = []
        rng = np.random.default_rng(Config.RANDOM_SEED)

        for uid in train_users:
            cands = candidates.get(uid)
            if cands is None or len(cands) == 0:
                continue
            if isinstance(cands, np.ndarray):
                cands = cands.tolist()
            cands = cands[:max_per_user]
            future = val_purchases.get(uid, set())

            pos_items = [item for item in cands if item in future]
            neg_items = [item for item in cands if item not in future]

            if len(pos_items) == 0:
                continue

            user_cats = {self.article_to_category.get(item) for item in user_hist_ctx.get(uid, set())}
            user_pcs = user_pcs_ctx.get(uid, set())

            hard_negs = [
                item for item in neg_items
                if self.article_to_category.get(item) in user_cats
                or self.article_to_product_code.get(item) in user_pcs
            ]
            easy_negs = [item for item in neg_items if item not in hard_negs]

            n_neg_total = min(len(neg_items), len(pos_items) * Config.NEG_POS_RATIO)
            n_hard = min(len(hard_negs), int(n_neg_total * Config.HARD_NEG_RATIO))
            n_easy = min(len(easy_negs), n_neg_total - n_hard)

            if len(hard_negs) > n_hard:
                hard_negs = list(rng.choice(hard_negs, size=n_hard, replace=False))
            if len(easy_negs) > n_easy:
                easy_negs = list(rng.choice(easy_negs, size=n_easy, replace=False))

            selected_negs = hard_negs + easy_negs
            item_to_rank = {item: idx for idx, item in enumerate(cands)}

            for item in pos_items:
                rows.append({
                    'customer_id': uid, 'article_id': item, 'label': 1,
                    'candidate_rank': item_to_rank[item]
                })
            for item in selected_negs:
                rows.append({
                    'customer_id': uid, 'article_id': item, 'label': 0,
                    'candidate_rank': item_to_rank.get(item, max_per_user - 1)
                })

        df_week = pd.DataFrame(rows)
        if len(df_week) == 0:
            return None

        df_week = self.build_features(df_week, context=temporal_ctx)
        return df_week

    def build_training_samples(self, candidates, train_users, max_per_user=100):
        print("\nConstructing time-lagged unbiased training dataset...")
        print("  -> Building Week 1 samples (target: 09-09 to 09-15, features up to 09-08)...")
        df_w1 = self._sample_for_week(
            pd.Timestamp('2020-09-09'), pd.Timestamp('2020-09-15'),
            candidates, train_users, max_per_user=max_per_user
        )

        print("  -> Building Week 2 samples (target: 09-02 to 09-08, features up to 09-01)...")
        df_w2 = self._sample_for_week(
            pd.Timestamp('2020-09-02'), pd.Timestamp('2020-09-08'),
            candidates, train_users, max_per_user=max_per_user
        )

        train_df = pd.concat([df_w1, df_w2], ignore_index=True)
        train_df = train_df.sort_values('customer_id').reset_index(drop=True)

        n_pos = train_df['label'].sum()
        n_neg = len(train_df) - n_pos
        n_users_with_pos = train_df[train_df['label'] == 1]['customer_id'].nunique()
        print(f"Unbiased total samples: {len(train_df):,} (pos: {n_pos:,}, neg: {n_neg:,}, ratio 1:{n_neg / max(n_pos, 1):.1f})")
        print(f"Unique users with positive labels: {n_users_with_pos:,}")

        drop_cols = {'customer_id', 'article_id', 'label', 'candidate_rank_bucket'}
        self.feature_cols = [c for c in train_df.columns if c not in drop_cols]
        print(f"Total active ranking features: {len(self.feature_cols)}")
        return train_df

    def train(self, train_df, seed=42):
        if train_df is None or len(train_df) == 0:
            return None

        self.models_ranker = []
        self.models_classifier = []

        np.random.seed(seed)
        unique_users = train_df['customer_id'].unique()
        val_users = set(np.random.choice(unique_users, size=int(len(unique_users) * 0.15), replace=False))

        train_mask = ~train_df['customer_id'].isin(val_users)
        val_mask = train_df['customer_id'].isin(val_users)

        train_sub = train_df[train_mask].copy().reset_index(drop=True)
        val_sub = train_df[val_mask].copy().reset_index(drop=True)

        X_train = train_sub[self.feature_cols]
        y_train = train_sub['label']
        X_val = val_sub[self.feature_cols]
        y_val = val_sub['label']

        train_groups = train_sub.groupby('customer_id', sort=False).size().values
        val_groups = val_sub.groupby('customer_id', sort=False).size().values

        for i, s in enumerate(Config.ENSEMBLE_SEEDS):
            print(f"\n{'=' * 50}")
            print(f"Training ensemble model {i + 1}/{len(Config.ENSEMBLE_SEEDS)} (seed={s})")
            print('=' * 50)

            print(f"  Fitting LGBMRanker (seed={s})...")
            model_r = lgb.LGBMRanker(
                objective="lambdarank",
                metric="map",
                n_estimators=Config.LGB_N_ESTIMATORS,
                learning_rate=Config.LGB_LEARNING_RATE,
                num_leaves=Config.LGB_NUM_LEAVES,
                max_depth=Config.LGB_MAX_DEPTH,
                min_child_samples=Config.LGB_MIN_CHILD_SAMPLES,
                subsample=Config.LGB_SUBSAMPLE,
                colsample_bytree=Config.LGB_COLSAMPLE_BYTREE,
                reg_alpha=Config.LGB_REG_ALPHA,
                reg_lambda=Config.LGB_REG_LAMBDA,
                random_state=s,
                verbose=-1,
                n_jobs=-1,
                lambdarank_truncation_level=Config.LGB_TRUNCATION_LEVEL,
                label_gain=Config.LGB_LABEL_GAIN,
            )
            model_r.fit(
                X_train, y_train,
                group=train_groups,
                eval_set=[(X_val, y_val)],
                eval_group=[val_groups],
                eval_at=[12],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            self.models_ranker.append(model_r)

            print(f"  Fitting LGBMClassifier (seed={s})...")
            model_c = lgb.LGBMClassifier(
                n_estimators=Config.LGB_N_ESTIMATORS,
                learning_rate=Config.LGB_LEARNING_RATE,
                num_leaves=Config.LGB_NUM_LEAVES,
                max_depth=Config.LGB_MAX_DEPTH,
                min_child_samples=Config.LGB_MIN_CHILD_SAMPLES,
                subsample=Config.LGB_SUBSAMPLE,
                colsample_bytree=Config.LGB_COLSAMPLE_BYTREE,
                reg_alpha=Config.LGB_REG_ALPHA,
                reg_lambda=Config.LGB_REG_LAMBDA,
                random_state=s,
                verbose=-1,
                n_jobs=-1,
            )
            model_c.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            self.models_classifier.append(model_c)

            val_pred_r = model_r.predict(X_val)
            val_pred_c = model_c.predict_proba(X_val)[:, 1]
            print(f"    seed={s} Ranker AUC: {roc_auc_score(y_val, val_pred_r):.4f}, "
                  f"Classifier AUC: {roc_auc_score(y_val, val_pred_c):.4f}")

        imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.models_ranker[0].feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 20 Feature Importances:")
        for i, row in imp.head(20).iterrows():
            print(f"  {row['feature']:<35} {row['importance']:>10.0f}")

        return self.models_ranker[0]

    def predict_batch(self, candidates, batch_size=5000, top_n=100, ensemble=True):
        """
        Generates predictions using dual-head RRF (k0=12) combined with smooth temporal decay priors.
        """
        recs = {}
        users = list(candidates.keys())

        for i in tqdm(range(0, len(users), batch_size), desc="LGBM prediction", leave=False):
            batch = users[i:i + batch_size]
            rows = []
            for uid in batch:
                cands = candidates.get(uid, [])[:top_n]
                for idx, item in enumerate(cands):
                    rows.append({
                        'customer_id': uid,
                        'article_id': item,
                        'candidate_rank': idx
                    })

            if not rows:
                continue

            df = pd.DataFrame(rows)
            df = self.build_features(df, context=self.default_context)

            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            if ensemble and len(self.models_ranker) > 0:
                scores_r = [m_r.predict(df[self.feature_cols]) for m_r in self.models_ranker]
                scores_c = [m_c.predict_proba(df[self.feature_cols])[:, 1] for m_c in self.models_classifier]

                df['score_r'] = np.mean(scores_r, axis=0)
                df['score_c'] = np.mean(scores_c, axis=0)

                df['rank_r'] = df.groupby('customer_id')['score_r'].rank(ascending=False)
                df['rank_c'] = df.groupby('customer_id')['score_c'].rank(ascending=False)

                # Internal dual-head RRF (k0=12)
                base_rrf = 0.50 / (12.0 + df['rank_r']) + 0.50 / (12.0 + df['rank_c'])

                # Temporal half-life continuous repurchase alignment
                decay_boost = np.clip(df['decay_repurchase_score'].values * 0.065, 0.0, 0.065)
                df['score'] = base_rrf + decay_boost
            else:
                df['score'] = self.models_ranker[0].predict(df[self.feature_cols])

            for uid, group in df.groupby('customer_id'):
                top_items = group.nlargest(top_n, 'score')['article_id'].tolist()
                recs[uid] = top_items

            del df

        return recs