"""
Offline evaluation and hyperparameter grid search sandbox.
Performs fine-grained grid search over RRF smoothing constants (k0), 
ensemble weights, and repurchase priors using cached ranking predictions.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.prepare_data import load_data, split_data


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


def main():
    print("=" * 65)
    print("Starting Offline Hyperparameter Grid Search (RRF Optimization)")
    print("=" * 65)

    cache_dir = PROJECT_ROOT / 'cache'
    lgb_pred_cache = cache_dir / 'preds_lgb_best.pkl'
    cb_pred_cache = cache_dir / 'preds_cb_best.pkl'

    with open(lgb_pred_cache, 'rb') as f:
        lgb_preds = pickle.load(f)
    with open(cb_pred_cache, 'rb') as f:
        cb_preds = pickle.load(f)

    articles, customers, transactions = load_data(data_dir=str(Config.DATA_DIR))
    train_data, test_data = split_data(transactions, sample_ratio=Config.SAMPLE_RATIO)
    test_truth = {uid: g['article_id'].tolist() for uid, g in test_data.groupby('customer_id')}
    test_users = list(test_truth.keys())

    max_date = train_data['t_dat'].max()
    w1_trans = train_data[train_data['t_dat'] >= max_date - pd.Timedelta(days=7)].copy()
    w2_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=14)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=7))]
    w3_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=21)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=14))]
    w4_trans = train_data[(train_data['t_dat'] >= max_date - pd.Timedelta(days=28)) & (train_data['t_dat'] < max_date - pd.Timedelta(days=21))]

    w1_trans['days_ago'] = (max_date - w1_trans['t_dat']).dt.days
    w1_trans['decay_weight'] = 0.85 ** w1_trans['days_ago']
    weekly_hot_decay_items = w1_trans.groupby('article_id')['decay_weight'].sum().nlargest(12).index.tolist()

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

    u_w1 = w1_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w2 = w2_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w3 = w3_trans.groupby('customer_id')['article_id'].apply(list).to_dict()
    u_w4 = w4_trans.groupby('customer_id')['article_id'].apply(list).to_dict()

    user_last_date = train_data.groupby('customer_id')['t_dat'].max()
    user_silent_days = ((max_date - user_last_date).dt.days).to_dict()

    # Parameter grid for k0 search and weight allocation
    grid = [
        {"k0": 14, "w_lgb": 0.50, "w_cb": 0.50, "rep_boost": 0.35},
        {"k0": 12, "w_lgb": 0.50, "w_cb": 0.50, "rep_boost": 0.35},
        {"k0": 10, "w_lgb": 0.50, "w_cb": 0.50, "rep_boost": 0.35},
        {"k0": 8,  "w_lgb": 0.50, "w_cb": 0.50, "rep_boost": 0.35},
        {"k0": 6,  "w_lgb": 0.50, "w_cb": 0.50, "rep_boost": 0.35},
        {"k0": 4,  "w_lgb": 0.50, "w_cb": 0.50, "rep_boost": 0.35},
        {"k0": 8,  "w_lgb": 0.55, "w_cb": 0.45, "rep_boost": 0.35},
        {"k0": 8,  "w_lgb": 0.45, "w_cb": 0.55, "rep_boost": 0.35},
    ]

    print("\nStarting evaluation across parameter grid...")
    best_score = 0.0
    best_cfg = None

    for idx, cfg in enumerate(grid):
        k0 = cfg["k0"]
        wl, wc = cfg["w_lgb"], cfg["w_cb"]
        rb = cfg["rep_boost"]

        fused = {}

        for uid in test_users:
            l_items = lgb_preds.get(uid, [])[:80]
            c_items = cb_preds.get(uid, [])[:80]

            score_dict = {}

            # Reciprocal Rank Fusion
            for r, it in enumerate(l_items):
                score_dict[it] = score_dict.get(it, 0.0) + wl / (float(k0) + r)
            for r, it in enumerate(c_items):
                score_dict[it] = score_dict.get(it, 0.0) + wc / (float(k0) + r)

            silent_days = user_silent_days.get(uid, 999)
            u_age = user_age_map.get(uid, 30)
            target_hot = age_hot_map.get(get_age_bin(u_age), weekly_hot_decay_items)

            p_w1 = u_w1.get(uid, [])
            p_w2 = u_w2.get(uid, [])
            p_w3 = u_w3.get(uid, [])
            p_w4 = u_w4.get(uid, [])

            if len(p_w1) > 0 or len(p_w2) > 0:
                for it in p_w1: score_dict[it] = score_dict.get(it, 0.0) + rb
                for it in p_w2:
                    if it not in p_w1: score_dict[it] = score_dict.get(it, 0.0) + (rb * 0.5)

                for bought in set(p_w1 + p_w2):
                    pc = art_to_pc.get(bought)
                    if pc and pc in pc_to_arts:
                        for sibling in pc_to_arts[pc]:
                            if sibling != bought: score_dict[sibling] = score_dict.get(sibling, 0.0) + (rb * 0.28)

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

            fused[uid] = final_list

        score = compute_map12(fused, test_truth)
        tag = "[BEST]" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_cfg = cfg

        desc = f"RRF(k0={k0}, LGB={wl:.2f}, CB={wc:.2f}, rep_boost={rb:.2f})"
        print(f"[{idx+1:02d}] {desc:<48} -> MAP@12: {score:.4f} {tag}")

    print("\n" + "=" * 65)
    print(f"Optimal Configuration: {best_cfg} | Peak MAP@12: {best_score:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()