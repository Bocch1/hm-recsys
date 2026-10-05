"""
Feature engineering module.
Constructs rolling time-decay statistics, customer profiles, article lifecycle representations,
interaction features, and cross-attribute mapping dictionaries.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_item_features(articles):
    """
    Encodes discrete categorical metadata for catalog items.
    """
    item_features = articles[[
        'article_id', 'product_type_name', 'product_group_name',
        'colour_group_name', 'department_name', 'product_code',
        'graphical_appearance_name', 'perceived_colour_value_name'
    ]].copy()

    cat_cols = [
        'product_type_name', 'product_group_name', 'colour_group_name',
        'department_name', 'graphical_appearance_name', 'perceived_colour_value_name'
    ]
    for col in cat_cols:
        if col in item_features.columns:
            le = LabelEncoder()
            item_features[f'{col}_code'] = le.fit_transform(item_features[col].astype(str))

    return item_features


def build_user_features(customers):
    """
    Encodes demographic and static profile features for customers.
    """
    user_features = customers[['customer_id', 'club_member_status', 'fashion_news_frequency', 'age']].copy()
    user_features['club_member_status'] = user_features['club_member_status'].fillna('UNKNOWN')
    user_features['fashion_news_frequency'] = user_features['fashion_news_frequency'].fillna('NONE')
    user_features['age'] = user_features['age'].fillna(user_features['age'].median())

    for col in ['club_member_status', 'fashion_news_frequency']:
        le = LabelEncoder()
        user_features[f'{col}_code'] = le.fit_transform(user_features[col].astype(str))

    return user_features


def build_time_window_features(train_data):
    """
    Computes rolling time-decay sales window counts and burst momentum ratios.
    Dynamically anchored to the latest transaction timestamp to prevent lookahead bias.
    """
    now = train_data['t_dat'].max()
    time_windows = {
        '1d': now - pd.Timedelta(days=1),
        '3d': now - pd.Timedelta(days=3),
        '7d': now - pd.Timedelta(days=7),
        '14d': now - pd.Timedelta(days=14),
        '30d': now - pd.Timedelta(days=30),
    }

    # Calculate item transaction volume within each rolling temporal window
    for win_name, win_start in time_windows.items():
        win_data = train_data[train_data['t_dat'] >= win_start]
        item_counts = win_data['article_id'].value_counts()
        train_data[f'item_count_{win_name}'] = train_data['article_id'].map(item_counts).fillna(0).astype(np.int32)

    # 3-day vs 7-day velocity ratio for detecting sudden breakout trends
    c3d = train_data['item_count_3d'].values
    c7d = train_data['item_count_7d'].values
    train_data['item_burst_ratio_3d_7d'] = (c3d * 7.0) / (c7d * 3.0 + 1e-5)

    # Cumulative transaction counts across the entire training slice
    item_counts_all = train_data['article_id'].value_counts()
    train_data['item_count_all'] = train_data['article_id'].map(item_counts_all).fillna(0).astype(np.int32)

    return train_data


def build_interaction_features(train_data, articles):
    """
    Constructs historical user-item and user-category interaction frequencies.
    """
    user_item_counts = train_data.groupby(['customer_id', 'article_id']).size().reset_index(name='user_item_count')
    train_data = train_data.merge(user_item_counts, on=['customer_id', 'article_id'], how='left')

    # Category affinity counts
    category_map = articles.set_index('article_id')['product_type_name'].to_dict()
    train_data['category'] = train_data['article_id'].map(category_map)
    user_cat_counts = train_data.groupby(['customer_id', 'category']).size().reset_index(name='user_cat_count')
    train_data = train_data.merge(user_cat_counts, on=['customer_id', 'category'], how='left')
    train_data.drop('category', axis=1, inplace=True)

    return train_data


def build_user_stats(train_data):
    """
    Constructs behavioral user profiling (purchasing power, repurchase propensity, and active span).
    """
    now = train_data['t_dat'].max()
    user_stats = train_data.groupby('customer_id').agg(
        user_buy_count=('article_id', 'count'),
        user_unique_items=('article_id', 'nunique'),
        user_first_purchase=('t_dat', 'min'),
        user_last_purchase=('t_dat', 'max'),
        user_avg_price=('price', 'mean'),
        user_std_price=('price', 'std'),
        user_max_price=('price', 'max'),
        user_min_price=('price', 'min'),
    ).reset_index()

    # User repurchase ratio (approaches 1.0 if user predominantly reorders familiar items)
    user_stats['user_repurchase_ratio'] = 1.0 - (user_stats['user_unique_items'] / user_stats['user_buy_count'])

    # Lifespan duration and recency interval
    user_stats['user_active_days'] = (user_stats['user_last_purchase'] - user_stats['user_first_purchase']).dt.days
    user_stats['user_days_since_last'] = (now - user_stats['user_last_purchase']).dt.days
    user_stats['user_std_price'] = user_stats['user_std_price'].fillna(0.0)

    return user_stats


def build_item_stats(train_data, user_features):
    """
    Constructs article profiling (lifecycle maturity, buyer age distribution, and velocity).
    """
    now = train_data['t_dat'].max()

    # Buyer demographic age distributions
    item_age_stats = train_data.merge(
        user_features[['customer_id', 'age']], on='customer_id', how='left'
    ).groupby('article_id')['age'].agg(['mean', 'min', 'max']).reset_index()
    item_age_stats.columns = ['article_id', 'item_mean_buyer_age', 'item_min_buyer_age', 'item_max_buyer_age']

    # Sales volume and transactional boundaries
    item_stats = train_data.groupby('article_id').agg(
        item_popularity=('customer_id', 'count'),
        item_avg_price=('price', 'mean'),
        item_first_purchase=('t_dat', 'min'),
        item_last_purchase=('t_dat', 'max'),
    ).reset_index()

    item_stats = item_stats.merge(item_age_stats, on='article_id', how='left')

    # Item catalog maturity and sales velocity
    item_stats['item_days_since_first'] = (now - item_stats['item_first_purchase']).dt.days
    item_stats['item_days_since_last'] = (now - item_stats['item_last_purchase']).dt.days
    item_stats['item_sales_velocity'] = item_stats['item_popularity'] / (item_stats['item_days_since_first'] + 1.0)

    return item_stats


def build_mappings(articles, train_data, item_stats):
    """
    Constructs lookup mappings for rapid vectorization and cross-attribute lookups.
    """
    article_to_category = articles.set_index('article_id')['product_type_name'].to_dict()
    article_to_product_code = articles.set_index('article_id')['product_code'].to_dict()
    article_to_dept = articles.set_index('article_id')['department_name'].to_dict() if 'department_name' in articles.columns else {}

    # Historical purchase lookup dictionaries
    user_purchased_hist = train_data.groupby('customer_id')['article_id'].apply(set).to_dict()
    user_purchased_list = train_data.groupby('customer_id')['article_id'].apply(list).to_dict()

    # Map historical purchases to product_code to track cross-color variants
    train_data_pc = train_data.copy()
    train_data_pc['product_code'] = train_data_pc['article_id'].map(article_to_product_code)
    user_product_codes_hist = train_data_pc.groupby('customer_id')['product_code'].apply(set).to_dict()

    item_popularity_map = item_stats.set_index('article_id')['item_popularity'].to_dict()
    article_avg_price_map = item_stats.set_index('article_id')['item_avg_price'].to_dict()

    # Category popularity ranking
    category_to_articles = {}
    for cat, group in articles.groupby('product_type_name'):
        items = group['article_id'].tolist()
        items_sorted = sorted(items, key=lambda x: item_popularity_map.get(x, 0), reverse=True)
        category_to_articles[cat] = items_sorted

    product_code_to_articles = {}
    for pc, group in articles.groupby('product_code'):
        product_code_to_articles[pc] = group['article_id'].tolist()

    return {
        'article_to_category': article_to_category,
        'article_to_product_code': article_to_product_code,
        'article_to_dept': article_to_dept,
        'article_avg_price_map': article_avg_price_map,
        'user_purchased_hist': user_purchased_hist,
        'user_purchased_list': user_purchased_list,
        'user_product_codes_hist': user_product_codes_hist,
        'category_to_articles': category_to_articles,
        'product_code_to_articles': product_code_to_articles,
    }


def run_all(articles, customers, train_data):
    """
    Executes the complete feature engineering pipeline.
    """
    print("=" * 60)
    print("3. Feature Engineering Pipeline")
    print("=" * 60)

    start_time = time.time()

    item_features = build_item_features(articles)
    user_features = build_user_features(customers)
    print("Categorical encoding completed.")

    train_data = build_time_window_features(train_data)
    print("Time-decay window statistics computed (1d/3d/7d/14d/30d).")

    train_data = build_interaction_features(train_data, articles)
    print("Interaction features generated.")

    user_stats = build_user_stats(train_data)
    print(f"Customer profiles generated: {len(user_stats):,} (including repurchase ratios and price dispersion).")

    item_stats = build_item_stats(train_data, user_features)
    print(f"Article profiles generated: {len(item_stats):,} (including lifecycles and sales velocities).")

    mappings = build_mappings(articles, train_data, item_stats)
    print("Entity mappings generated (including cross-color product_code mappings).")

    print(f"Elapsed time: {time.time() - start_time:.2f}s")
    print("Feature engineering tables constructed successfully.\n")

    return item_features, user_features, train_data, user_stats, item_stats, mappings