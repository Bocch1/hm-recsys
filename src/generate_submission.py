"""
Kaggle full submission pipeline (chunked batch streaming inference for 1.37M users).
Applies RRF (k0=12) ensemble ranking, seasonal decay priors, and demographic fallbacks.
"""

import sys
import time
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier

from config import Config
from src.data.prepare_data import load_data
from src.features.build_features import run_all
from src.features.image_text_embedding import ImageTextEmbeddingExtractor
from src.rank.embedding_ranker import EmbeddingRanker


def main():
    start = time.time()
    print("=" * 65)
    print("Generating Kaggle Submission (1,371,980 Customers)")
    print("=" * 65)

    cache_dir = PROJECT_ROOT / 'cache'
    output_dir = PROJECT_ROOT / 'submissions'
    output_dir.mkdir(exist_ok=True)
    submission_path = output_dir / 'submission.csv'

    print("\n1. Loading datasets...")
    articles, customers, transactions = load_data(data_dir=str(Config.DATA_DIR))

    print("\n2. Computing features and embeddings...")
    if Config.USE_RECENT_MONTHS > 0:
        recent = transactions['t_dat'].max() - pd.Timedelta(days=Config.USE_RECENT_MONTHS * 30)
        train_data = transactions[transactions['t_dat'] >= recent].copy()
    else:
        train_data = transactions.copy()

    item_features, user_features, train_data, user_stats, item_stats, mappings = run_all(
        articles, customers, train_data
    )

    images_dir = Config.DATA_DIR / 'images'
    extractor = ImageTextEmbeddingExtractor(articles, images_dir=images_dir)
    image_emb, text_emb = extractor.extract_all(dim=32)

    # Compute time-decay purchase statistics and demographic trending items
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

    def get_age_bin(age):
        if age < 25: return 0
        elif age < 35: return 1
        elif age < 45: return 2
        elif age < 55: return 3
        else: return 4

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

    # Load precomputed active customer predictions
    lgb_pred_cache = cache_dir / 'preds_lgb_best.pkl'
    cb_pred_cache = cache_dir / 'preds_cb_best.pkl'

    lgb_preds = {}
    cb_preds = {}
    if lgb_pred_cache.exists() and cb_pred_cache.exists():
        print("\nLoading precomputed model predictions from cache...")
        with open(lgb_pred_cache, 'rb') as f:
            lgb_preds = pickle.load(f)
        with open(cb_pred_cache, 'rb') as f:
            cb_preds = pickle.load(f)

    # Stream predictions directly to CSV
    print(f"\n3. Streaming output -> {submission_path}...")
    all_customer_ids = customers['customer_id'].values
    k0 = 12.0
    rep_boost = 0.35

    with open(submission_path, 'w', encoding='utf-8') as f_out:
        f_out.write("customer_id,prediction\n")

        for uid in tqdm(all_customer_ids, desc="Generating submission predictions"):
            l_items = lgb_preds.get(uid)
            c_items = cb_preds.get(uid)

            silent_days = user_silent_days.get(uid, 999)
            u_age = user_age_map.get(uid, 30)
            target_hot = age_hot_map.get(get_age_bin(u_age), weekly_hot_decay_items)

            # Scenario A: Customer has model ranking scores available
            if l_items is not None and c_items is not None:
                score_dict = {}
                for r, it in enumerate(l_items[:80]):
                    score_dict[it] = score_dict.get(it, 0.0) + 0.50 / (k0 + r)
                for r, it in enumerate(c_items[:80]):
                    score_dict[it] = score_dict.get(it, 0.0) + 0.50 / (k0 + r)

                p_w1 = u_w1.get(uid, [])
                p_w2 = u_w2.get(uid, [])
                p_w3 = u_w3.get(uid, [])
                p_w4 = u_w4.get(uid, [])

                if len(p_w1) > 0 or len(p_w2) > 0:
                    for it in p_w1: score_dict[it] = score_dict.get(it, 0.0) + rep_boost
                    for it in p_w2:
                        if it not in p_w1: score_dict[it] = score_dict.get(it, 0.0) + (rep_boost * 0.5)

                    for bought in set(p_w1 + p_w2):
                        pc = art_to_pc.get(bought)
                        if pc and pc in pc_to_arts:
                            for sibling in pc_to_arts[pc]:
                                if sibling != bought: score_dict[sibling] = score_dict.get(sibling, 0.0) + (rep_boost * 0.28)

                    for rank, item in enumerate(target_hot):
                        score_dict[item] = score_dict.get(item, 0.0) + 0.05 * (1.0 - rank / 12.0)
                else:
                    for it in p_w3: score_dict[it] = score_dict.get(it, 0.0) + 0.10
                    for it in p_w4: score_dict[it] = score_dict.get(it, 0.0) + 0.05

                    hw = 0.20 if silent_days > 28 else 0.08
                    for rank, item in enumerate(target_hot):
                        score_dict[item] = score_dict.get(item, 0.0) + hw * (1.0 - rank / 12.0)

                sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
                final_list = [it for it, _ in sorted_items[:12]]
                if len(final_list) < 12:
                    for it in target_hot:
                        if it not in final_list: final_list.append(it)
                        if len(final_list) == 12: break

            # Scenario B: Cold-start or dormant user (demographic & trending fallback)
            else:
                p_w1 = u_w1.get(uid, [])
                p_w2 = u_w2.get(uid, [])
                final_list = []
                for it in (p_w1 + p_w2):
                    if it not in final_list: final_list.append(it)
                    if len(final_list) == 12: break

                for it in target_hot:
                    if len(final_list) >= 12: break
                    if it not in final_list: final_list.append(it)

            pred_str = " ".join([str(x) for x in final_list[:12]])
            f_out.write(f"{uid},{pred_str}\n")

    print(f"\nSubmission generated successfully: {submission_path}")
    print(f"Total time elapsed: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()