"""
End-to-end training, prediction, and evaluation pipeline.
Integrates feature engineering, multi-channel candidate recall, dual-head LightGBM, 
CatBoost classifier, and metric-aligned RRF fusion.
"""

import sys
import time
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier

from config import Config
from src.data.prepare_data import load_data, split_data
from src.features.build_features import run_all
from src.features.image_text_embedding import ImageTextEmbeddingExtractor
from src.recall.award_recall import AwardRecall
from src.rank.embedding_ranker import EmbeddingRanker


def compute_map12(recommendations, test_truth_dict):
    """Compute Mean Average Precision at 12 (MAP@12)."""
    aps = []
    for uid, true_items in test_truth_dict.items():
        if uid not in recommendations:
            aps.append(0.0)
            continue
        rec_list = recommendations[uid][:12]
        true_set = set(true_items)
        m, n = len(true_items), len(rec_list)
        if m == 0 or n == 0:
            aps.append(0.0)
            continue
        norm = 1.0 / min(m, 12)
        hits, ps = 0, 0.0
        for k in range(1, min(n, 12) + 1):
            if rec_list[k - 1] in true_set:
                hits += 1
                ps += hits / k
        aps.append(norm * ps)
    return np.mean(aps) if aps else 0.0


def get_age_bin(age):
    """Discretize customer age into demographic cohorts."""
    if age < 25: return 0
    elif age < 35: return 1
    elif age < 45: return 2
    elif age < 55: return 3
    else: return 4


def reprioritize_candidates(raw_candidates, train_data, articles, customers, max_cands=100, split_name="Candidate Pool"):
    """
    Hierarchical candidate reprioritization:
    4-week repurchase ladder -> active trending autumn items -> department/age cohorts -> original recall backfill.
    """
    print(f"\nExecuting candidate reprioritization [{split_name}] (Top-{max_cands})...")
    max_date = train_data['t_dat'].max()

    w1_trans = train_data[train_data['t_dat'] >= max_date - pd.Timedelta(days=7)].copy()
    w2_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=14)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=7))]
    w3_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=21)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=14))]
    w4_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=28)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=21))]

    w1_trans['days_ago'] = (max_date - w1_trans['t_dat']).dt.days
    w1_trans['decay_weight'] = 0.85 ** w1_trans['days_ago']
    weekly_hot_decay_40 = w1_trans.groupby('article_id')['decay_weight'].sum().nlargest(40).index.tolist()

    art_to_dept = articles.set_index('article_id')['department_name'].to_dict()
    art_to_pc = articles.set_index('article_id')['product_code'].to_dict()
    pc_to_arts = articles.groupby('product_code')['article_id'].apply(list).to_dict()
    art_to_index_group = articles.set_index('article_id')['index_group_name'].to_dict()

    user_age_map = customers.set_index('customer_id')['age'].fillna(30).to_dict()

    w1_trans['dept'] = w1_trans['article_id'].map(art_to_dept)
    w1_trans['index_group'] = w1_trans['article_id'].map(art_to_index_group)
    w1_trans['age_bin'] = w1_trans['customer_id'].map(user_age_map).apply(get_age_bin)

    dept_hot_map = {}
    for d, grp in w1_trans.groupby('dept'):
        dept_hot_map[d] = grp.groupby('article_id')['decay_weight'].sum().nlargest(8).index.tolist()

    age_hot_map = {}
    for abin, grp in w1_trans.groupby('age_bin'):
        age_hot_map[abin] = grp.groupby('article_id')['decay_weight'].sum().nlargest(10).index.tolist()

    index_hot_map = {}
    for igrp, grp in w1_trans.groupby('index_group'):
        index_hot_map[igrp] = grp.groupby('article_id')['decay_weight'].sum().nlargest(10).index.tolist()

    recent_60d = train_data[train_data['t_dat'] >= max_date - pd.Timedelta(days=60)][['customer_id', 'article_id']].copy()
    recent_60d['dept'] = recent_60d['article_id'].map(art_to_dept)
    recent_60d['index_group'] = recent_60d['article_id'].map(art_to_index_group)

    user_top_depts = recent_60d.groupby(['customer_id', 'dept']).size().unstack(fill_value=0)
    user_fav_depts = {uid: list(row.nlargest(2).index) for uid, row in user_top_depts.iterrows()}

    user_top_index = recent_60d.groupby(['customer_id', 'index_group']).size().unstack(fill_value=0)
    user_fav_index = user_top_index.idxmax(axis=1).to_dict() if len(user_top_index) > 0 else {}
    del recent_60d, user_top_depts, user_top_index

    u_w1 = w1_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w2 = w2_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w3 = w3_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w4 = w4_trans.groupby('customer_id')['article_id'].apply(list).to_dict()

    new_candidates = {}

    for uid, cands in tqdm(raw_candidates.items(), desc=f"Reprioritizing {split_name}", leave=False):
        prioritized = []
        seen = set()

        def add_item(it):
            if it not in seen:
                prioritized.append(it)
                seen.add(it)

        # 1. Repurchase candidates across recent 4 weeks
        for it in u_w1.get(uid, []): add_item(it)
        for it in u_w2.get(uid, []): add_item(it)

        for it in (u_w1.get(uid, []) + u_w2.get(uid, [])):
            pc = art_to_pc.get(it)
            if pc and pc in pc_to_arts:
                for sibling in pc_to_arts[pc]: add_item(sibling)

        for it in u_w3.get(uid, []): add_item(it)
        for it in u_w4.get(uid, []): add_item(it)

        # 2. Trending autumn items in current week
        for it in weekly_hot_decay_40: add_item(it)

        # 3. Department and demographic trending candidates
        for d in user_fav_depts.get(uid, []):
            for it in dept_hot_map.get(d, []): add_item(it)

        u_abin = get_age_bin(user_age_map.get(uid, 30))
        for it in age_hot_map.get(u_abin, []): add_item(it)

        u_igrp = user_fav_index.get(uid, 'Ladieswear')
        for it in index_hot_map.get(u_igrp, []): add_item(it)

        # 4. Backfill from original retrieval candidates
        for it in cands:
            add_item(it)
            if len(prioritized) >= max_cands:
                break

        new_candidates[uid] = prioritized[:max_cands]

    return new_candidates


def main():
    start = time.time()
    Config.print_config()

    cache_dir = PROJECT_ROOT / 'cache'
    cache_dir.mkdir(exist_ok=True)

    articles, customers, transactions = load_data(data_dir=str(Config.DATA_DIR))

    train_data, test_data = split_data(transactions, sample_ratio=Config.SAMPLE_RATIO)
    if Config.USE_RECENT_MONTHS > 0:
        recent = train_data['t_dat'].max() - pd.Timedelta(days=Config.USE_RECENT_MONTHS * 30)
        train_data = train_data[train_data['t_dat'] >= recent].copy()
        print(f"Using recent {Config.USE_RECENT_MONTHS} months: {len(train_data):,} records")

    item_features, user_features, train_data, user_stats, item_stats, mappings = run_all(
        articles, customers, train_data
    )

    images_dir = Config.DATA_DIR / 'images'
    extractor = ImageTextEmbeddingExtractor(articles, images_dir=images_dir)
    image_emb, text_emb = extractor.extract_all(dim=32)

    # ============================================================
    # 4. Candidate Retrieval & Reprioritization
    # ============================================================
    print("\n4. Loading candidate pools and applying reprioritization...")
    train_cand_path = cache_dir / 'train_candidates.pkl'
    test_cand_path = cache_dir / 'test_candidates.pkl'
    train_df_path = cache_dir / 'train_features.pkl'



    # Load candidates or trigger recall engine automatically
    test_users = test_data['customer_id'].unique()
    all_train_users = train_data['customer_id'].unique()
    train_users_subset = list(all_train_users[:Config.RANK_TRAIN_USERS])

    if train_cand_path.exists() and test_cand_path.exists():
        print(f"-> Loading cached candidate pools ({train_cand_path.name}, {test_cand_path.name})...")
        with open(train_cand_path, 'rb') as f:
            train_candidates = pickle.load(f)
        with open(test_cand_path, 'rb') as f:
            test_candidates = pickle.load(f)
    else:
        print("-> Candidate cache not found. Generating candidates via AwardRecall...")
        recaller = AwardRecall(train_data, articles, customers, mappings)

        print(f"  Retrieving candidates for {len(train_users_subset):,} training customers...")
        train_candidates = recaller.recall_batch(train_users_subset, top_k=200)

        print(f"  Retrieving candidates for {len(test_users):,} validation customers...")
        test_candidates = recaller.recall_batch(test_users, top_k=200)

        with open(cache_dir / 'train_candidates.pkl', 'wb') as f:
            pickle.dump(train_candidates, f)
        with open(cache_dir / 'test_candidates.pkl', 'wb') as f:
            pickle.dump(test_candidates, f)
        print("  Candidate pools generated and cached.")

    train_candidates_100 = reprioritize_candidates(train_candidates, train_data, articles, customers, max_cands=100, split_name="Train Set")
    test_candidates_100 = reprioritize_candidates(test_candidates, train_data, articles, customers, max_cands=100, split_name="Test Set")

    train_users = [u for u in train_candidates_100.keys()][:Config.RANK_TRAIN_USERS]

    # ============================================================
    # 5. Model 1: LightGBM Ranker Training
    # ============================================================
    print("\n5. Training primary ranker (Dual-head LightGBM)...")
    ranker_lgb = EmbeddingRanker(
        train_data, articles, customers, mappings,
        item_similarity_dict={},
        image_emb=image_emb, text_emb=text_emb,
    )

    if train_df_path.exists():
        print(f"-> Loading cached training features ({train_df_path.name})...")
        train_df = pd.read_pickle(train_df_path)
        ranker_lgb.feature_cols = [col for col in train_df.columns if col not in ['customer_id', 'article_id', 'label', 'candidate_rank_bucket']]
    else:
        print("-> Building training features from scratch...")
        train_df = ranker_lgb.build_training_samples(
            train_candidates_100, train_users, max_per_user=100
        )
        train_df.to_pickle(cache_dir / 'train_features.pkl')

    ranker_lgb.train(train_df, seed=42)

    # ============================================================
    # 6. Model 2: CatBoost Classifier Training
    # ============================================================
    print("\n6. Training auxiliary model (CatBoost Classifier)...")
    feature_cols = ranker_lgb.feature_cols

    np.random.seed(42)
    unique_users = train_df['customer_id'].unique()
    val_users = np.random.choice(unique_users, size=int(len(unique_users) * 0.15), replace=False)
    val_mask = train_df['customer_id'].isin(set(val_users))
    train_mask = ~val_mask

    X_train = train_df[train_mask][feature_cols].fillna(0)
    y_train = train_df[train_mask]['label']
    X_val = train_df[val_mask][feature_cols].fillna(0)
    y_val = train_df[val_mask]['label']

    cat_model = CatBoostClassifier(
        iterations=450,
        learning_rate=0.045,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        thread_count=16
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=35,
        use_best_model=True
    )
    print("CatBoost model training completed.")

    # ============================================================
    # 7. Test Predictions & Ranking Extraction
    # ============================================================
    print("\n7. Generating test predictions and extracting rankings...")
    test_truth = {uid: g['article_id'].tolist() for uid, g in test_data.groupby('customer_id')}

    print("-> Generating LightGBM Top-100 predictions (Dual-head RRF + temporal prior)...")
    lgb_recs_100 = ranker_lgb.predict_batch(
        test_candidates_100,
        batch_size=Config.PREDICT_BATCH_SIZE,
        top_n=100
    )

    print("-> Generating CatBoost Top-100 predictions (Temporal prior)...")
    all_test_rows = []
    for uid in tqdm(test_users, desc="Building test features for CatBoost", leave=False):
        cands = test_candidates_100.get(uid, [])
        for idx, item in enumerate(cands):
            all_test_rows.append({
                'customer_id': uid,
                'article_id': item,
                'candidate_rank': idx
            })

    test_df = pd.DataFrame(all_test_rows)
    test_df = ranker_lgb.build_features(test_df, context=ranker_lgb.default_context)

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    test_df['cb_prob'] = cat_model.predict_proba(test_df[feature_cols].fillna(0))[:, 1]
    test_df['cb_rank'] = test_df.groupby('customer_id')['cb_prob'].rank(ascending=False)
    cb_base_rrf = 1.0 / (12.0 + test_df['cb_rank'])
    cb_decay_boost = np.clip(test_df['decay_repurchase_score'].values * 0.065, 0.0, 0.065)
    test_df['cb_final_score'] = cb_base_rrf + cb_decay_boost

    cb_recs_100 = {}
    cb_pure_top12 = {}
    for uid, group in test_df.groupby('customer_id'):
        sorted_items = group.nlargest(100, 'cb_final_score')['article_id'].tolist()
        cb_recs_100[uid] = sorted_items
        cb_pure_top12[uid] = sorted_items[:12]

    del test_df

    with open(cache_dir / 'preds_lgb_best.pkl', 'wb') as f:
        pickle.dump(lgb_recs_100, f)
    with open(cache_dir / 'preds_cb_best.pkl', 'wb') as f:
        pickle.dump(cb_recs_100, f)
    print("Rank predictions saved to cache/preds_lgb_best.pkl and cache/preds_cb_best.pkl.")

    # ============================================================
    # 8. Ensemble Fusion: Reciprocal Rank Fusion (k0=12)
    # ============================================================
    print("\n8. Executing ensemble fusion (RRF k0=12)...")
    max_date = train_data['t_dat'].max()
    w1_trans = train_data[train_data['t_dat'] >= max_date - pd.Timedelta(days=7)].copy()
    w2_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=14)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=7))]
    w3_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=21)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=14))]
    w4_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=28)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=21))]

    w1_trans['days_ago'] = (max_date - w1_trans['t_dat']).dt.days
    w1_trans['decay_weight'] = 0.85 ** w1_trans['days_ago']
    weekly_hot_decay_items = w1_trans.groupby('article_id')['decay_weight'].sum().nlargest(12).index.tolist()

    u_w1 = w1_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w2 = w2_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w3 = w3_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w4 = w4_trans.groupby('customer_id')['article_id'].apply(list).to_dict()

    art_to_pc = articles.set_index('article_id')['product_code'].to_dict()
    pc_to_arts = articles.groupby('product_code')['article_id'].apply(list).to_dict()

    user_age_map = customers.set_index('customer_id')['age'].fillna(30).to_dict()
    w1_with_age = w1_trans.copy()
    w1_with_age['age_bin'] = w1_with_age['customer_id'].map(user_age_map).apply(get_age_bin)

    age_hot_map = {}
    for abin in range(5):
        sub_items = w1_with_age[w1_with_age['age_bin'] == abin].groupby('article_id')['decay_weight'].sum().nlargest(12).index.tolist()
        if len(sub_items) < 12:
            sub_items += [it for it in weekly_hot_decay_items if it not in sub_items][:12 - len(sub_items)]
        age_hot_map[abin] = sub_items

    user_last_date = train_data.groupby('customer_id')['t_dat'].max()
    user_silent_days = ((max_date - user_last_date).dt.days).to_dict()

    fused_recs = {}
    lgb_pure_top12 = {}

    k0 = 12.0
    rep_boost = 0.35

    for uid in tqdm(test_users, desc="Computing RRF ensemble recommendations"):
        lgb_items = lgb_recs_100.get(uid, [])
        cb_items = cb_recs_100.get(uid, [])
        lgb_pure_top12[uid] = lgb_items[:12]

        score_dict = {}

        for rank, item in enumerate(lgb_items[:80]):
            score_dict[item] = score_dict.get(item, 0.0) + 0.50 / (k0 + rank)

        for rank, item in enumerate(cb_items[:80]):
            score_dict[item] = score_dict.get(item, 0.0) + 0.50 / (k0 + rank)

        silent_days = user_silent_days.get(uid, 999)
        u_age = user_age_map.get(uid, 30)
        u_age_bin = get_age_bin(u_age)
        target_hot_items = age_hot_map.get(u_age_bin, weekly_hot_decay_items)

        p_w1 = u_w1.get(uid, [])
        p_w2 = u_w2.get(uid, [])
        p_w3 = u_w3.get(uid, [])
        p_w4 = u_w4.get(uid, [])

        has_recent_repurchase = len(p_w1) > 0 or len(p_w2) > 0

        if has_recent_repurchase:
            for it in p_w1: score_dict[it] = score_dict.get(it, 0.0) + rep_boost
            for it in p_w2:
                if it not in p_w1: score_dict[it] = score_dict.get(it, 0.0) + (rep_boost * 0.5)

            for bought_item in set(p_w1 + p_w2):
                pc = art_to_pc.get(bought_item)
                if pc and pc in pc_to_arts:
                    for sibling in pc_to_arts[pc]:
                        if sibling != bought_item:
                            score_dict[sibling] = score_dict.get(sibling, 0.0) + (rep_boost * 0.28)

            for rank, item in enumerate(target_hot_items):
                score_dict[item] = score_dict.get(item, 0.0) + 0.05 * (1.0 - rank / 12.0)

        else:
            for it in p_w3: score_dict[it] = score_dict.get(it, 0.0) + 0.10
            for it in p_w4: score_dict[it] = score_dict.get(it, 0.0) + 0.05

            hw = 0.20 if silent_days > 28 else 0.08
            for rank, item in enumerate(target_hot_items):
                score_dict[item] = score_dict.get(item, 0.0) + hw * (1.0 - rank / 12.0)

        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

        final_list = []
        for item, _ in sorted_items:
            if item not in final_list:
                final_list.append(item)
            if len(final_list) == 12:
                break

        for item in target_hot_items:
            if len(final_list) >= 12:
                break
            if item not in final_list:
                final_list.append(item)

        fused_recs[uid] = final_list[:12]

    # ============================================================
    # 9. Evaluation Summary
    # ============================================================
    map12_lgb = compute_map12(lgb_pure_top12, test_truth)
    map12_cb = compute_map12(cb_pure_top12, test_truth)
    map12_fused = compute_map12(fused_recs, test_truth)

    print(f"\n{'=' * 65}")
    print(f"Evaluation Results Summary")
    print(f"{'=' * 65}")
    print(f"{'Model / Approach':<42} {'MAP@12':<12}")
    print("-" * 55)
    print(f"{'LightGBM Single Model (Dual-Head RRF)':<42} {map12_lgb:<12.4f}")
    print(f"{'CatBoost Single Model (Temporal Prior)':<42} {map12_cb:<12.4f}")
    print(f"{'Final Ensemble (RRF k0=12)':<42} {map12_fused:<12.4f}")
    print(f"Total pipeline time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()