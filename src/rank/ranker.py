"""
Ranking model base class with multi-stage hard negative mining.

Key Architecture:
1. Multi-stage training pipeline:
   - Stage 1: Train baseline ranker via random negative sampling.
   - Stage 2: Mine hard negatives (false positives scored highly by the base model).
   - Stage 3: Retrain final model using a balanced mixture of hard, category-matched, and random negatives.
2. Negative mixture composition: 40% model-mined hard, 30% category-matched, 30% random negatives.
3. Multi-seed ensemble for variance reduction.
4. Comprehensive user, item, interaction, streak, and window-based feature engineering.
"""

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import Config


class Ranker:
    """Base ranking engine with hard negative mining capabilities."""

    def __init__(self, train_data, articles, customers, mappings, item_similarity_dict=None):
        self.train_data = train_data
        self.articles = articles
        self.customers = customers
        self.mappings = mappings
        self.item_similarity_dict = item_similarity_dict or {}

        self.article_to_category = mappings['article_to_category']
        self.article_to_department = dict(zip(articles['article_id'], articles['department_no']))
        self.article_to_section = dict(zip(articles['article_id'], articles['section_no']))
        self.article_to_colour = dict(zip(articles['article_id'], articles['colour_group_code']))
        self.article_to_product_type = dict(zip(articles['article_id'], articles['product_type_no']))
        self.user_purchased_hist = mappings['user_purchased_hist']

        self._build_all_features()
        self.feature_cols = []
        self.models_ranker = []
        self.models_classifier = []

    # ============================================================
    # Feature Construction
    # ============================================================
    def _build_all_features(self):
        print("Constructing ranking features...")
        self._build_user_features()
        self._build_item_features()
        self._build_user_item_interactions()
        self._build_age_distribution()
        self._build_repurchase_features()
        self._build_streak_features()
        self._build_time_window_features()
        print("Feature precomputation completed.")

    def _build_user_features(self):
        c = self.customers
        self.user_age = dict(zip(c['customer_id'], c['age'].fillna(0).astype(np.float32)))
        self.user_FN = dict(zip(c['customer_id'], c['FN'].fillna(0).astype(np.float32)))
        self.user_Active = dict(zip(c['customer_id'], c['Active'].fillna(0).astype(np.float32)))

        club_map = {'ACTIVE': 1, 'PRE-CREATE': 0, 'LEFT CLUB': -1}
        self.user_club_status = dict(zip(
            c['customer_id'],
            c['club_member_status'].map(club_map).fillna(0).astype(np.float32)
        ))

        freq_map = {'Regularly': 3, 'Monthly': 2, 'None': 0, 'Seasonally': 1}
        self.user_fashion_freq = dict(zip(
            c['customer_id'],
            c['fashion_news_frequency'].map(freq_map).fillna(0).astype(np.float32)
        ))

        postal = c['postal_code'].fillna('UNKNOWN').values
        self.user_postal_code = dict(zip(
            c['customer_id'].values,
            np.array([float(hash(str(p)) % 100000) for p in postal], dtype=np.float32)
        ))

        user_stats = self.train_data.groupby('customer_id').agg(
            total=('article_id', 'count'),
            first=('t_dat', 'min'),
            last=('t_dat', 'max')
        )
        user_stats['days'] = (user_stats['last'] - user_stats['first']).dt.days.clip(lower=1)
        user_stats['freq'] = user_stats['total'] / user_stats['days']
        self.user_purchase_freq = user_stats['freq'].to_dict()
        self.user_total_purchases = user_stats['total'].to_dict()

    def _build_item_features(self):
        df = self.train_data
        self.item_purchase_count = df.groupby('article_id').size().to_dict()
        self.item_purchase_users = df.groupby('article_id')['customer_id'].nunique().to_dict()

        item_time = df.groupby('article_id')['t_dat'].agg(['min', 'max'])
        now = df['t_dat'].max()
        self.item_days_since_last = {a: (now - t).days for a, t in item_time['max'].items()}
        self.item_days_since_first = {a: (now - t).days for a, t in item_time['min'].items()}

        if 'sales_channel_id' in df.columns:
            self.item_sales_channel = df.groupby('article_id')['sales_channel_id'].mean().to_dict()
        else:
            self.item_sales_channel = defaultdict(lambda: 1.0)

    def _build_user_item_interactions(self):
        df = self.train_data.copy()
        df['category'] = df['article_id'].map(self.article_to_category)
        df['department'] = df['article_id'].map(self.article_to_department)
        df['colour'] = df['article_id'].map(self.article_to_colour)

        self.user_category_count = df.groupby(['customer_id', 'category']).size().to_dict()
        self.user_colour_count = df.groupby(['customer_id', 'colour']).size().to_dict()
        self.user_department_count = df.groupby(['customer_id', 'department']).size().to_dict()

        now = df['t_dat'].max()
        user_cat_last = df.groupby(['customer_id', 'category'])['t_dat'].max().to_dict()
        self.user_category_last_days = {(u, c): (now - t).days for (u, c), t in user_cat_last.items()}

        self.user_category_avg_interval = {}
        for (u, c), g in df.groupby(['customer_id', 'category']):
            if len(g) > 1:
                times = g['t_dat'].sort_values()
                intervals = [(times.iloc[i + 1] - times.iloc[i]).days for i in range(len(times) - 1)]
                self.user_category_avg_interval[(u, c)] = float(np.mean(intervals))
            else:
                self.user_category_avg_interval[(u, c)] = 0.0

    def _build_age_distribution(self):
        df = self.train_data.merge(
            self.customers[['customer_id', 'age']], on='customer_id', how='left'
        )
        df['age'] = df['age'].fillna(0)
        df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 100], labels=False)

        self.item_age_group_count = df.groupby(['article_id', 'age_group']).size().to_dict()
        self.item_avg_age = df.groupby('article_id')['age'].mean().to_dict()

        item_total = df.groupby('article_id').size().to_dict()
        self.item_age_ratio = {
            (a, g): cnt / item_total.get(a, 1)
            for (a, g), cnt in self.item_age_group_count.items()
        }

    def _build_repurchase_features(self):
        df = self.train_data
        user_p = df.groupby('customer_id')['article_id'].agg(['count', 'nunique'])
        self.user_repurchase_rate = (1 - user_p['nunique'] / user_p['count']).to_dict()

        item_p = df.groupby('article_id')['customer_id'].agg(['count', 'nunique'])
        self.item_repurchase_rate = (1 - item_p['nunique'] / item_p['count']).to_dict()

        user_t = df.groupby('customer_id')['t_dat'].agg(['min', 'max', 'count'])
        user_t['avg'] = (user_t['max'] - user_t['min']).dt.days / (user_t['count'] - 1).clip(lower=1)
        self.user_avg_interval = user_t['avg'].to_dict()

        item_t = df.groupby('article_id')['t_dat'].agg(['min', 'max', 'count'])
        item_t['avg'] = (item_t['max'] - item_t['min']).dt.days / (item_t['count'] - 1).clip(lower=1)
        self.item_avg_interval = item_t['avg'].to_dict()

        user_t['months'] = ((user_t['max'] - user_t['min']).dt.days / 30).clip(lower=1)
        self.user_monthly_freq = (user_t['count'] / user_t['months']).to_dict()

    def _build_streak_features(self):
        df = self.train_data.sort_values(['customer_id', 't_dat'])
        self.user_category_streak = {}
        self.user_department_streak = {}
        self.user_section_streak = {}
        self.user_item_streak = {}

        for uid, g in tqdm(df.groupby('customer_id'), desc="  Computing streaks", leave=False):
            items = g['article_id'].tail(50).tolist()
            cats = [self.article_to_category.get(i, 0) for i in items]
            depts = [self.article_to_department.get(i, 0) for i in items]
            sections = [self.article_to_section.get(i, 0) for i in items]

            for key, seq, target in [
                ('cat', cats, self.user_category_streak),
                ('dept', depts, self.user_department_streak),
                ('section', sections, self.user_section_streak),
                ('item', items, self.user_item_streak)
            ]:
                streak = 1
                for i in range(len(seq) - 1, 0, -1):
                    if seq[i] == seq[i - 1]:
                        streak += 1
                    else:
                        break
                target[uid] = float(streak)

    def _build_time_window_features(self):
        now = self.train_data['t_dat'].max()
        self.item_window_stats = {}
        for name, days in [('1d', 1), ('3d', 3), ('7d', 7), ('30d', 30), ('90d', 90)]:
            w = self.train_data[self.train_data['t_dat'] >= now - pd.Timedelta(days=days)]
            self.item_window_stats[name] = {
                'user_count': w.groupby('article_id')['customer_id'].nunique().to_dict(),
                'purchase_count': w['article_id'].value_counts().to_dict(),
            }

        self.item_trend = {}
        for a in self.item_window_stats['90d']['purchase_count']:
            c90 = self.item_window_stats['90d']['purchase_count'].get(a, 0)
            self.item_trend[a] = {
                '7d': self.item_window_stats['7d']['purchase_count'].get(a, 0) / max(c90 / 13, 1),
                '3d': self.item_window_stats['3d']['purchase_count'].get(a, 0) / max(c90 / 30, 1),
                '1d': self.item_window_stats['1d']['purchase_count'].get(a, 0) / max(c90 / 90, 1),
            }

    # ============================================================
    # Feature Vectorization
    # ============================================================
    def build_features(self, df):
        cids = df['customer_id'].values
        aids = df['article_id'].values

        df['user_age'] = np.array([self.user_age.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_FN'] = np.array([self.user_FN.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_Active'] = np.array([self.user_Active.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_club_status'] = np.array([self.user_club_status.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_fashion_freq'] = np.array([self.user_fashion_freq.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_postal_code'] = np.array([self.user_postal_code.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_purchase_freq'] = np.array([self.user_purchase_freq.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_total_purchases'] = np.array([self.user_total_purchases.get(u, 0.0) for u in cids], dtype=np.float32)

        df['item_purchase_count'] = np.array([self.item_purchase_count.get(a, 0) for a in aids], dtype=np.float32)
        df['item_purchase_users'] = np.array([self.item_purchase_users.get(a, 0) for a in aids], dtype=np.float32)
        df['item_days_since_last'] = np.array([self.item_days_since_last.get(a, 999) for a in aids], dtype=np.float32)
        df['item_days_since_first'] = np.array([self.item_days_since_first.get(a, 999) for a in aids], dtype=np.float32)
        df['item_sales_channel'] = np.array([self.item_sales_channel.get(a, 1.0) for a in aids], dtype=np.float32)

        df['item_trend_7d'] = np.array([self.item_trend.get(a, {}).get('7d', 0) for a in aids], dtype=np.float32)
        df['item_trend_3d'] = np.array([self.item_trend.get(a, {}).get('3d', 0) for a in aids], dtype=np.float32)
        df['item_trend_1d'] = np.array([self.item_trend.get(a, {}).get('1d', 0) for a in aids], dtype=np.float32)

        cat_series = df['article_id'].map(self.article_to_category)
        if cat_series.dtype == object:
            unique_cats = {c: i for i, c in enumerate(cat_series.dropna().unique())}
            df['_cat'] = cat_series.map(unique_cats).fillna(-1).astype(np.int32)
        else:
            df['_cat'] = cat_series.fillna(-1).astype(np.int32)

        df['user_category_count'] = np.array([
            self.user_category_count.get((u, c), 0) for u, c in zip(cids, df['_cat'].values)
        ], dtype=np.float32)
        df['user_category_last_days'] = np.array([
            self.user_category_last_days.get((u, c), 999) for u, c in zip(cids, df['_cat'].values)
        ], dtype=np.float32)
        df['user_category_avg_interval'] = np.array([
            self.user_category_avg_interval.get((u, c), 0) for u, c in zip(cids, df['_cat'].values)
        ], dtype=np.float32)
        df['time_match_score'] = (df['user_category_last_days'] - df['user_category_avg_interval']).clip(-100, 100)

        col_series = df['article_id'].map(self.article_to_colour).fillna(-1).astype(np.int32)
        df['user_colour_count'] = np.array([
            self.user_colour_count.get((u, c), 0) for u, c in zip(cids, col_series.values)
        ], dtype=np.float32)

        dept_series = df['article_id'].map(self.article_to_department).fillna(-1).astype(np.int32)
        df['user_department_count'] = np.array([
            self.user_department_count.get((u, d), 0) for u, d in zip(cids, dept_series.values)
        ], dtype=np.float32)

        df['item_avg_age'] = np.array([self.item_avg_age.get(a, 0.0) for a in aids], dtype=np.float32)
        df['_age_group'] = pd.cut(df['user_age'], bins=[0, 20, 30, 40, 50, 60, 100], labels=False).fillna(0).astype(np.int32)
        df['item_age_group_count'] = np.array([
            self.item_age_group_count.get((a, g), 0) for a, g in zip(aids, df['_age_group'].values)
        ], dtype=np.float32)
        df['item_age_ratio'] = np.array([
            self.item_age_ratio.get((a, g), 0) for a, g in zip(aids, df['_age_group'].values)
        ], dtype=np.float32)
        df['age_match'] = np.abs(df['user_age'] - df['item_avg_age'])

        df['user_repurchase_rate'] = np.array([self.user_repurchase_rate.get(u, 0.0) for u in cids], dtype=np.float32)
        df['item_repurchase_rate'] = np.array([self.item_repurchase_rate.get(a, 0.0) for a in aids], dtype=np.float32)
        df['user_avg_interval'] = np.array([self.user_avg_interval.get(u, 0.0) for u in cids], dtype=np.float32)
        df['item_avg_interval'] = np.array([self.item_avg_interval.get(a, 0.0) for a in aids], dtype=np.float32)
        df['user_monthly_freq'] = np.array([self.user_monthly_freq.get(u, 0.0) for u in cids], dtype=np.float32)
        df['repurchase_cycle_match'] = np.abs(df['user_avg_interval'] - df['item_avg_interval'])

        df['user_category_streak'] = np.array([self.user_category_streak.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_department_streak'] = np.array([self.user_department_streak.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_section_streak'] = np.array([self.user_section_streak.get(u, 0.0) for u in cids], dtype=np.float32)
        df['user_item_streak'] = np.array([self.user_item_streak.get(u, 0.0) for u in cids], dtype=np.float32)

        for name in ['1d', '3d', '7d', '30d', '90d']:
            df[f'item_user_count_{name}'] = np.array([
                self.item_window_stats[name]['user_count'].get(a, 0) for a in aids
            ], dtype=np.float32)
            df[f'item_purchase_count_{name}'] = np.array([
                self.item_window_stats[name]['purchase_count'].get(a, 0) for a in aids
            ], dtype=np.float32)

        if self.item_similarity_dict:
            df['avg_item_similarity'] = np.array([
                self._compute_similarity(u, a) for u, a in zip(cids, aids)
            ], dtype=np.float32)

        if 'candidate_rank' in df.columns:
            df['candidate_rank_bucket'] = pd.cut(
                df['candidate_rank'],
                bins=[-1, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
                labels=False
            ).fillna(9).astype(np.float32)
            df = df.drop('candidate_rank', axis=1, errors='ignore')

        df['user_age_x_category'] = (df['_age_group'] * 1000 + df['_cat']).astype(np.float32)

        if 'candidate_rank_bucket' in df.columns:
            df['cat_count_x_rank_bucket'] = (df['user_category_count'] * df['candidate_rank_bucket']).astype(np.float32)
            df['trend_x_rank_bucket'] = (df['item_trend_7d'] * (10 - df['candidate_rank_bucket'])).astype(np.float32)
            df['repurchase_x_rank_bucket'] = (df['item_repurchase_rate'] * df['candidate_rank_bucket']).astype(np.float32)

        df = df.drop(['_cat', '_age_group'], axis=1, errors='ignore')
        return df

    def _compute_similarity(self, uid, aid):
        purchased = list(self.user_purchased_hist.get(uid, set()))[:5]
        if not purchased:
            return 0.0
        scores = []
        for item in purchased:
            for sim_item, score in self.item_similarity_dict.get(item, [])[:5]:
                if sim_item == aid:
                    scores.append(float(score))
                    break
        return float(np.mean(scores)) if scores else 0.0

    # ============================================================
    # Multi-Stage Hard Negative Mining
    # ============================================================
    def _build_candidate_records(self, candidates, train_users, val_purchases, max_per_user=500):
        """Constructs candidate pair dataframes preserving rank and binary labels."""
        records = []
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

            item_to_rank = {item: idx for idx, item in enumerate(cands)}

            for item in pos_items:
                records.append({
                    'customer_id': uid, 'article_id': item, 'label': 1,
                    'candidate_rank': item_to_rank[item]
                })
            for item in neg_items:
                records.append({
                    'customer_id': uid, 'article_id': item, 'label': 0,
                    'candidate_rank': item_to_rank[item]
                })
        return pd.DataFrame(records)

    def stage1_train_base_model(self, candidates, train_users, seed=42):
        """
        Stage 1: Train baseline ranker via uniform random negative sampling.
        """
        print("\n" + "=" * 60)
        print("Stage 1: Training Baseline Ranker via Negative Downsampling")
        print("=" * 60)

        val_start = pd.Timestamp(Config.PSEUDO_VAL_START)
        val_end = pd.Timestamp(Config.PSEUDO_VAL_END)
        val_data = self.train_data[
            (self.train_data['t_dat'] >= val_start) &
            (self.train_data['t_dat'] < val_end)
        ]
        val_purchases = val_data.groupby('customer_id')['article_id'].apply(set).to_dict()

        all_records_df = self._build_candidate_records(
            candidates, train_users, val_purchases, max_per_user=Config.PREDICT_TOP_N
        )
        print(f"Total candidate records: {len(all_records_df):,}")

        pos_df = all_records_df[all_records_df['label'] == 1]
        neg_df = all_records_df[all_records_df['label'] == 0]

        pos_counts = pos_df.groupby('customer_id').size()
        sampled_negs = []
        for uid, group in neg_df.groupby('customer_id'):
            n_pos = pos_counts.get(uid, 0)
            if n_pos == 0:
                continue
            n_neg = min(len(group), n_pos * Config.NEG_POS_RATIO)
            sampled = group.sample(n=n_neg, random_state=seed)
            sampled_negs.append(sampled)
        neg_sampled = pd.concat(sampled_negs) if sampled_negs else pd.DataFrame()

        base_df = pd.concat([pos_df, neg_sampled]).reset_index(drop=True)
        print(f"Stage 1 training samples: {len(base_df):,} (pos: {len(pos_df):,}, neg: {len(neg_sampled):,})")

        base_df = self.build_features(base_df)
        feature_cols = [c for c in base_df.columns if c not in ['customer_id', 'article_id', 'label']]

        X = base_df[feature_cols]
        y = base_df['label']

        np.random.seed(seed)
        users = base_df['customer_id'].unique()
        val_users = set(np.random.choice(users, size=int(len(users) * 0.15), replace=False))
        val_mask = base_df['customer_id'].isin(val_users)
        train_mask = ~val_mask

        base_model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="map",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
            lambdarank_truncation_level=20,
        )
        base_model.fit(
            X[train_mask], y[train_mask],
            group=base_df[train_mask].groupby('customer_id').size().values,
            eval_set=[(X[val_mask], y[val_mask])],
            eval_group=[base_df[val_mask].groupby('customer_id').size().values],
            eval_at=[12],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        self.base_model = base_model
        self.base_feature_cols = feature_cols
        print("Stage 1 completed. Baseline model successfully trained.")

        return base_model

    def stage2_mine_hard_negatives(self, candidates, train_users, top_k=50):
        """
        Stage 2: Mine hard negatives using the baseline model.
        Identifies negative items predicted with top-k scores by the stage 1 model.
        """
        print("\n" + "=" * 60)
        print("Stage 2: Mining Hard Negative Candidates")
        print("=" * 60)

        val_start = pd.Timestamp(Config.PSEUDO_VAL_START)
        val_end = pd.Timestamp(Config.PSEUDO_VAL_END)
        val_data = self.train_data[
            (self.train_data['t_dat'] >= val_start) &
            (self.train_data['t_dat'] < val_end)
        ]
        val_purchases = val_data.groupby('customer_id')['article_id'].apply(set).to_dict()

        all_records_df = self._build_candidate_records(
            candidates, train_users, val_purchases, max_per_user=Config.PREDICT_TOP_N
        )

        all_records_df = self.build_features(all_records_df)
        all_records_df['base_score'] = self.base_model.predict(
            all_records_df[self.base_feature_cols]
        )

        hard_negs = {}
        for uid, group in all_records_df.groupby('customer_id'):
            negs = group[group['label'] == 0]
            if len(negs) == 0:
                continue
            top_hard = negs.nlargest(top_k, 'base_score')['article_id'].tolist()
            hard_negs[uid] = set(top_hard)

        total_hard = sum(len(v) for v in hard_negs.values())
        print(f"Hard negatives mined: {total_hard:,} (avg {total_hard / max(len(hard_negs), 1):.1f} per user)")

        return hard_negs

    def stage3_train_final(self, candidates, train_users, hard_negs, seed=42):
        """
        Stage 3: Train final ranking ensemble using balanced negative mixture.
        """
        print("\n" + "=" * 60)
        print("Stage 3: Training Final Ranking Model on Hard Negative Mixture")
        print("=" * 60)

        val_start = pd.Timestamp(Config.PSEUDO_VAL_START)
        val_end = pd.Timestamp(Config.PSEUDO_VAL_END)
        val_data = self.train_data[
            (self.train_data['t_dat'] >= val_start) &
            (self.train_data['t_dat'] < val_end)
        ]
        val_purchases = val_data.groupby('customer_id')['article_id'].apply(set).to_dict()

        rows = []
        rng = np.random.default_rng(seed)
        n_users_with_pos = 0

        for uid in tqdm(train_users, desc="  Constructing final training dataset", leave=False):
            cands = candidates.get(uid)
            if cands is None or len(cands) == 0:
                continue
            if isinstance(cands, np.ndarray):
                cands = cands.tolist()
            cands = cands[:Config.PREDICT_TOP_N]
            future = val_purchases.get(uid, set())

            pos_items = [item for item in cands if item in future]
            neg_items = [item for item in cands if item not in future]

            if len(pos_items) == 0:
                continue

            n_users_with_pos += 1

            hard_set = hard_negs.get(uid, set())
            user_cats = set()
            for item in self.user_purchased_hist.get(uid, set()):
                cat = self.article_to_category.get(item)
                if cat is not None:
                    user_cats.add(cat)

            model_hard = [item for item in neg_items if item in hard_set]
            cat_hard = [item for item in neg_items
                        if item not in hard_set
                        and self.article_to_category.get(item) in user_cats]
            random_neg = [item for item in neg_items
                          if item not in hard_set
                          and self.article_to_category.get(item) not in user_cats]

            n_neg_total = min(len(neg_items), len(pos_items) * Config.NEG_POS_RATIO)

            n_model_hard = min(len(model_hard), int(n_neg_total * 0.4))
            n_cat_hard = min(len(cat_hard), int(n_neg_total * 0.3))
            n_random = min(len(random_neg), n_neg_total - n_model_hard - n_cat_hard)

            if len(model_hard) > n_model_hard:
                model_hard = list(rng.choice(model_hard, size=n_model_hard, replace=False))
            if len(cat_hard) > n_cat_hard:
                cat_hard = list(rng.choice(cat_hard, size=n_cat_hard, replace=False))
            if len(random_neg) > n_random:
                random_neg = list(rng.choice(random_neg, size=n_random, replace=False))

            selected_negs = model_hard + cat_hard + random_neg

            item_to_rank = {item: idx for idx, item in enumerate(cands)}

            for item in pos_items:
                rows.append({
                    'customer_id': uid, 'article_id': item, 'label': 1,
                    'candidate_rank': item_to_rank[item]
                })
            for item in selected_negs:
                rows.append({
                    'customer_id': uid, 'article_id': item, 'label': 0,
                    'candidate_rank': item_to_rank.get(item, Config.PREDICT_TOP_N - 1)
                })

        final_df = pd.DataFrame(rows)
        n_pos = final_df['label'].sum()
        n_neg = len(final_df) - n_pos
        print(f"Stage 3 training records: {len(final_df):,} (pos: {n_pos:,}, neg: {n_neg:,}, ratio 1:{n_neg / max(n_pos, 1):.1f})")
        print(f"Users with positive interactions: {n_users_with_pos:,}")

        final_df = self.build_features(final_df)
        self.feature_cols = [c for c in final_df.columns if c not in ['customer_id', 'article_id', 'label']]
        print(f"Total features: {len(self.feature_cols)}")

        self._train_final_models(final_df, seed)
        return final_df

    def _train_final_models(self, train_df, seed=42):
        """Trains final ensemble rankers and classifiers across seeds."""
        self.models_ranker = []
        self.models_classifier = []

        for i, s in enumerate(Config.ENSEMBLE_SEEDS):
            print(f"\n  Training final ensemble {i + 1}/{len(Config.ENSEMBLE_SEEDS)} (seed={s})")

            np.random.seed(s)
            users = train_df['customer_id'].unique()
            val_users = set(np.random.choice(users, size=int(len(users) * 0.15), replace=False))
            val_mask = train_df['customer_id'].isin(val_users)
            train_mask = ~val_mask

            X_train = train_df[train_mask][self.feature_cols]
            y_train = train_df[train_mask]['label']
            X_val = train_df[val_mask][self.feature_cols]
            y_val = train_df[val_mask]['label']

            train_groups = train_df[train_mask].groupby('customer_id').size().values
            val_groups = train_df[val_mask].groupby('customer_id').size().values

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

        print(f"\nTop 25 Feature Importances:")
        for i, row in imp.head(25).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>10.0f}")

    # ============================================================
    # Batch Inference
    # ============================================================
    def predict_batch(self, candidates, batch_size=5000, top_n=500, ensemble=True):
        recs = {}
        users = list(candidates.keys())

        for i in tqdm(range(0, len(users), batch_size), desc="Batch inference", leave=False):
            batch = users[i:i + batch_size]
            rows = []
            for uid in batch:
                cands = candidates.get(uid)
                if cands is None or len(cands) == 0:
                    continue
                if isinstance(cands, np.ndarray):
                    cands = cands.tolist()
                cands = cands[:top_n]
                for idx, item in enumerate(cands):
                    rows.append({'customer_id': uid, 'article_id': item, 'candidate_rank': idx})

            if not rows:
                continue

            df = pd.DataFrame(rows)
            df = self.build_features(df)

            if ensemble and len(self.models_ranker) > 0:
                scores_r = []
                scores_c = []
                for model_r, model_c in zip(self.models_ranker, self.models_classifier):
                    sr = model_r.predict(df[self.feature_cols])
                    sr = (sr - sr.min()) / (sr.max() - sr.min() + 1e-6)
                    scores_r.append(sr)

                    sc = model_c.predict_proba(df[self.feature_cols])[:, 1]
                    sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-6)
                    scores_c.append(sc)

                avg_r = np.mean(scores_r, axis=0)
                avg_c = np.mean(scores_c, axis=0)
                df['score'] = (Config.ENSEMBLE_WEIGHT_RANKER * avg_r
                               + Config.ENSEMBLE_WEIGHT_CLASSIFIER * avg_c)
            else:
                df['score'] = self.models_ranker[0].predict(df[self.feature_cols])

            for uid, group in df.groupby('customer_id'):
                top12 = group.nlargest(12, 'score')['article_id'].tolist()
                recs[uid] = top12

            del df

        return recs