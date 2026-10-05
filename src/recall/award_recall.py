"""
Multi-channel high-coverage candidate retrieval engine.

Retrieval Channels:
1. Repurchase ladder (recent purchases)
2. Sibling variants (shared product_code)
3. In-basket item co-occurrence
4. Sequential next-item transition patterns
5. Demographic age-group popularity
6. User-User Collaborative Filtering
7. Sales channel popularity
8. Item-based Collaborative Filtering (ItemCF)
9. Graph embedding representation (Word2Vec / Item2Vec)
10. Global trending items fallback
"""

from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import Config


# ============================================================
# Sub-Retrievers
# ============================================================

class GraphEmbeddingRecall:
    """Graph embedding retrieval based on Item2Vec skip-gram representation."""

    def __init__(self, train_data, user_purchased_list):
        self.train_data = train_data
        self.user_purchased_list = user_purchased_list
        self.embedding_dim = 32
        self.article_embedding = {}

    def fit(self):
        from gensim.models import Word2Vec
        print("  Fitting Graph Embedding (Item2Vec)...")
        sequences = []
        for uid, items in self.user_purchased_list.items():
            if len(items) >= 3:
                sequences.append([str(item) for item in items])
        if len(sequences) < 100:
            return {}
        model = Word2Vec(
            sentences=sequences, vector_size=self.embedding_dim,
            window=3, min_count=2, workers=4, epochs=5, sg=1,
        )
        for item_str in model.wv.index_to_key:
            self.article_embedding[int(item_str)] = model.wv[item_str]
        for item in self.article_embedding:
            norm = np.linalg.norm(self.article_embedding[item])
            if norm > 0:
                self.article_embedding[item] = self.article_embedding[item] / norm
        print(f"  Graph Embedding completed: {len(self.article_embedding):,} items")
        return self.article_embedding

    def recall(self, uid, purchased, top_k=100):
        if not purchased or not self.article_embedding:
            return []
        user_vector = None
        count = 0
        for item in list(purchased)[-10:]:
            if item in self.article_embedding:
                emb = self.article_embedding[item]
                user_vector = emb.copy() if user_vector is None else user_vector + emb
                count += 1
        if user_vector is None:
            return []
        user_vector = user_vector / count
        all_items = list(self.article_embedding.keys())
        all_vectors = np.array([self.article_embedding[i] for i in all_items])
        scores = all_vectors.dot(user_vector)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        results = []
        for idx in top_indices:
            item_id = all_items[idx]
            if item_id not in purchased:
                results.append(item_id)
                if len(results) >= top_k:
                    break
        return results


class CoOccurrenceRecall:
    """Basket co-occurrence candidate retrieval."""

    def __init__(self, train_data):
        self.train_data = train_data
        self.cooccurrence_dict = {}

    def fit(self):
        print("  Fitting Basket Co-occurrence...")
        recent = self.train_data[self.train_data['t_dat'] >= pd.Timestamp('2020-07-01')]
        recent = recent.sort_values(['customer_id', 't_dat'])
        recent['date'] = recent['t_dat'].dt.date
        baskets = recent.groupby(['customer_id', 'date'])['article_id'].apply(list).reset_index()
        cooc = defaultdict(lambda: defaultdict(int))
        for basket in tqdm(baskets['article_id'].values[:30000], desc="  Co-occurrence", leave=False):
            if len(basket) < 2:
                continue
            for i in range(min(len(basket), 10)):
                for j in range(i + 1, min(len(basket), 10)):
                    item1, item2 = basket[i], basket[j]
                    cooc[item1][item2] += 1
                    cooc[item2][item1] += 1
        for item, co_items in cooc.items():
            total = sum(co_items.values())
            self.cooccurrence_dict[item] = sorted(
                [(ci, cnt / total) for ci, cnt in co_items.items()],
                key=lambda x: x[1], reverse=True
            )[:30]
        print(f"  Basket Co-occurrence completed: {len(self.cooccurrence_dict):,} items")

    def recall(self, uid, purchased, top_k=100):
        if not purchased:
            return []
        candidates = defaultdict(float)
        for item in list(purchased)[-10:]:
            if item in self.cooccurrence_dict:
                for ci, score in self.cooccurrence_dict[item]:
                    if ci not in purchased:
                        candidates[ci] += score
        sorted_c = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_c[:top_k]]


class SequenceRecall:
    """Sequential transition pattern retrieval."""

    def __init__(self, train_data):
        self.train_data = train_data
        self.next_item_dict = {}

    def fit(self):
        print("  Fitting Sequential Patterns...")
        df = self.train_data.sort_values(['customer_id', 't_dat'])
        next_count = defaultdict(lambda: defaultdict(int))
        for uid, group in tqdm(df.groupby('customer_id'), desc="  Sequence transitions", leave=False):
            items = group['article_id'].tolist()
            for i in range(len(items) - 1):
                next_count[items[i]][items[i + 1]] += 1
        for item, next_items in next_count.items():
            total = sum(next_items.values())
            self.next_item_dict[item] = sorted(
                [(ni, cnt / total) for ni, cnt in next_items.items()],
                key=lambda x: x[1], reverse=True
            )[:30]
        print(f"  Sequential Patterns completed: {len(self.next_item_dict):,} items")

    def recall(self, uid, purchased, top_k=100):
        if not purchased:
            return []
        candidates = defaultdict(float)
        for item in list(purchased)[-3:]:
            if item in self.next_item_dict:
                for ni, score in self.next_item_dict[item]:
                    if ni not in purchased:
                        candidates[ni] += score
        sorted_c = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_c[:top_k]]


class AgeGroupPopularRecall:
    """Demographic age cohort popularity retrieval."""

    def __init__(self, train_data, customers):
        self.train_data = train_data
        self.customers = customers
        self.age_group_purchases = {}
        self.user_age = {}

    def fit(self):
        print("  Fitting Demographic Age Popularity...")
        self.user_age = dict(zip(self.customers['customer_id'], self.customers['age'].fillna(0)))
        df = self.train_data.merge(
            self.customers[['customer_id', 'age']], on='customer_id', how='left'
        )
        df['age_group'] = pd.cut(df['age'].fillna(0), bins=[0, 20, 30, 40, 50, 60, 100], labels=False)
        age_grouped = df.groupby(['age_group', 'article_id']).size().reset_index(name='count')
        for ag in age_grouped['age_group'].unique():
            data = age_grouped[age_grouped['age_group'] == ag]
            self.age_group_purchases[ag] = data.nlargest(1500, 'count')['article_id'].tolist()
        print(f"  Demographic Age Popularity completed: {len(self.age_group_purchases)} cohorts")

    def recall(self, uid, purchased, top_k=100):
        age = self.user_age.get(uid, 0)
        ag = pd.cut([age], bins=[0, 20, 30, 40, 50, 60, 100], labels=False)[0]
        if ag in self.age_group_purchases:
            return [item for item in self.age_group_purchases[ag]
                    if item not in purchased][:top_k]
        return []


class UserCFRecall:
    """User-User Collaborative Filtering retrieval."""

    def __init__(self, train_data, top_k_users=5000):
        self.train_data = train_data
        self.top_k_users = top_k_users
        self.user_similarity = {}
        self.user_items = {}

    def fit(self):
        print("  Fitting User-User Collaborative Filtering...")
        user_counts = self.train_data.groupby('customer_id').size()
        active_users = user_counts[user_counts >= 5].index.tolist()
        if len(active_users) > self.top_k_users:
            active_users = user_counts.loc[active_users].nlargest(self.top_k_users).index.tolist()

        df = self.train_data[self.train_data['customer_id'].isin(active_users)].copy()
        users = df['customer_id'].unique()
        items = df['article_id'].unique()
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {a: i for i, a in enumerate(items)}

        grouped = df.groupby(['customer_id', 'article_id']).size().reset_index(name='count')
        grouped['user_idx'] = grouped['customer_id'].map(user_idx)
        grouped['item_idx'] = grouped['article_id'].map(item_idx)

        matrix = csr_matrix(
            (grouped['count'].values, (grouped['user_idx'].values, grouped['item_idx'].values)),
            shape=(len(users), len(items))
        )

        sim = cosine_similarity(matrix, dense_output=False)
        sim.setdiag(0)

        for i, u in enumerate(users):
            row = sim.getrow(i)
            if row.nnz == 0:
                continue
            sorted_idx = np.argsort(row.data)[::-1][:50]
            self.user_similarity[u] = [
                (users[row.indices[j]], row.data[j])
                for j in sorted_idx if row.data[j] > 0.05
            ]

        for u in users:
            self.user_items[u] = set(df[df['customer_id'] == u]['article_id'].tolist())

        print(f"  User-User Collaborative Filtering completed: {len(self.user_similarity):,} users")

    def recall(self, uid, purchased, top_k=100):
        if uid not in self.user_similarity:
            return []
        candidates = defaultdict(float)
        for sim_uid, sim_score in self.user_similarity[uid][:20]:
            for item in self.user_items.get(sim_uid, set()):
                if item not in purchased:
                    candidates[item] += sim_score
        sorted_c = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_c[:top_k]]


class SalesChannelPopularRecall:
    """Sales channel popularity retrieval."""

    def __init__(self, train_data, customers):
        self.train_data = train_data
        self.customers = customers
        self.channel_items = {}
        self.user_channel = {}

    def fit(self):
        if 'sales_channel_id' not in self.train_data.columns:
            print("  Skipping sales channel retrieval (sales_channel_id column missing).")
            return

        print("  Fitting Sales Channel Popularity...")
        recent = self.train_data[self.train_data['t_dat'] >= pd.Timestamp('2020-07-01')]
        for ch in recent['sales_channel_id'].unique():
            data = recent[recent['sales_channel_id'] == ch]
            top_items = data['article_id'].value_counts().head(1000).index.tolist()
            self.channel_items[ch] = top_items

        user_ch = self.train_data.groupby('customer_id')['sales_channel_id'].agg(
            lambda x: x.mode().iloc[0] if len(x) > 0 else 2
        ).to_dict()
        self.user_channel = user_ch

        print(f"  Sales Channel Popularity completed: {len(self.channel_items)} channels")

    def recall(self, uid, purchased, top_k=100):
        if not self.channel_items:
            return []
        ch = self.user_channel.get(uid, 2)
        if ch in self.channel_items:
            return [item for item in self.channel_items[ch] if item not in purchased][:top_k]
        return []


# ============================================================
# Main Multi-Channel Retrieval Engine
# ============================================================

class AwardRecall:
    """Unified multi-channel high-coverage candidate retrieval engine."""

    def __init__(self, train_data, articles, customers, mappings):
        self.train_data = train_data
        self.articles = articles
        self.customers = customers
        self.mappings = mappings

        self.article_to_category = mappings['article_to_category']
        self.article_to_product_code = mappings['article_to_product_code']
        self.product_code_to_articles = mappings['product_code_to_articles']
        self.user_purchased_hist = mappings['user_purchased_hist']
        self.user_purchased_list = mappings['user_purchased_list']

        self.hot_items = train_data['article_id'].value_counts().head(Config.HOT_ITEMS_ALL).index.tolist()
        print(f"  Globally indexed trending items: {len(self.hot_items):,}")

        # ItemCF channel
        if Config.ENABLE_ITEMCF:
            print("  Building ItemCF...")
            self.item_similarity_dict = self._build_itemcf()
        else:
            self.item_similarity_dict = {}

        # Graph Embedding channel
        if Config.ENABLE_GRAPH:
            self.graph_recaller = GraphEmbeddingRecall(train_data, self.user_purchased_list)
            self.graph_recaller.fit()
        else:
            self.graph_recaller = None

        # Co-occurrence channel
        if Config.ENABLE_COOCCURRENCE:
            self.cooccurrence_recaller = CoOccurrenceRecall(train_data)
            self.cooccurrence_recaller.fit()
        else:
            self.cooccurrence_recaller = None

        # Sequential pattern channel
        if Config.ENABLE_SEQUENCE:
            self.sequence_recaller = SequenceRecall(train_data)
            self.sequence_recaller.fit()
        else:
            self.sequence_recaller = None

        # Demographic age channel
        if Config.ENABLE_AGE_GROUP:
            self.age_group_recaller = AgeGroupPopularRecall(train_data, customers)
            self.age_group_recaller.fit()
        else:
            self.age_group_recaller = None

        # User-User CF channel
        if Config.ENABLE_USER_CF:
            self.user_cf_recaller = UserCFRecall(train_data, Config.USER_CF_TOP_K)
            self.user_cf_recaller.fit()
        else:
            self.user_cf_recaller = None

        # Sales channel popularity
        if Config.ENABLE_SALES_CHANNEL:
            self.sales_channel_recaller = SalesChannelPopularRecall(train_data, customers)
            self.sales_channel_recaller.fit()
        else:
            self.sales_channel_recaller = None

        print("Multi-channel retrieval engine initialized successfully.")

    def _build_itemcf(self):
        hot = set(self.train_data['article_id'].value_counts().head(Config.ITEMCF_HOT_ITEMS).index.tolist())
        df = self.train_data[self.train_data['article_id'].isin(hot)]
        df = df[df['t_dat'] >= pd.Timestamp('2020-07-01')]
        if len(df) == 0:
            return {}

        users = df['customer_id'].unique()
        items = df['article_id'].unique()
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {a: i for i, a in enumerate(items)}

        grouped = df.groupby(['customer_id', 'article_id']).size().reset_index(name='count')
        grouped['user_idx'] = grouped['customer_id'].map(user_idx)
        grouped['item_idx'] = grouped['article_id'].map(item_idx)

        matrix = csr_matrix(
            (grouped['count'].values, (grouped['user_idx'].values, grouped['item_idx'].values)),
            shape=(len(users), len(items))
        )
        sim = cosine_similarity(matrix.T, dense_output=False)
        sim.setdiag(0)

        result = {}
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz == 0:
                continue
            item_id = items[i]
            sorted_idx = np.argsort(row.data)[::-1][:Config.ITEMCF_SIM_ITEMS]
            result[item_id] = [
                (items[row.indices[j]], row.data[j])
                for j in sorted_idx if row.data[j] > 0.01
            ]
        print(f"  ItemCF completed: {len(result):,} indexed items")
        return result

    def recall(self, uid, top_k=None):
        """Generates candidates for an individual customer across all active channels."""
        if top_k is None:
            top_k = Config.RECALL_TOP_K

        purchased = self.user_purchased_hist.get(uid, set())
        purchased_list = self.user_purchased_list.get(uid, [])

        candidates = []
        seen = set()

        def add(item):
            if item not in seen and item not in purchased:
                candidates.append(item)
                seen.add(item)

        # 1. Repurchase ladder
        last_basket = purchased_list[-Config.REPURCHASE_CANDIDATES:] if purchased_list else []
        for item in last_basket:
            if item not in seen:
                candidates.append(item)
                seen.add(item)

        # 2. Sibling variants (cross-color shared product_code)
        for item in last_basket[:10]:
            pc = self.article_to_product_code.get(item)
            if pc and pc in self.product_code_to_articles:
                for same_pc in self.product_code_to_articles[pc][:20]:
                    add(same_pc)

        # 3. In-basket item co-occurrence
        if self.cooccurrence_recaller:
            cooc = self.cooccurrence_recaller.recall(uid, purchased, top_k=Config.COOCCURRENCE_CANDIDATES)
            for item in cooc:
                add(item)

        # 4. Sequential next-item transitions
        if self.sequence_recaller:
            seq = self.sequence_recaller.recall(uid, purchased, top_k=Config.SEQUENCE_CANDIDATES)
            for item in seq:
                add(item)

        # 5. Demographic age cohort popularity
        if self.age_group_recaller:
            age = self.age_group_recaller.recall(uid, purchased, top_k=Config.AGE_GROUP_CANDIDATES)
            for item in age:
                add(item)

        # 6. User-User Collaborative Filtering
        if self.user_cf_recaller:
            user_cf_items = self.user_cf_recaller.recall(uid, purchased, top_k=Config.USER_CF_CANDIDATES)
            for item in user_cf_items:
                add(item)

        # 7. Preferred sales channel popularity
        if self.sales_channel_recaller:
            ch_items = self.sales_channel_recaller.recall(uid, purchased, top_k=Config.SALES_CHANNEL_CANDIDATES)
            for item in ch_items:
                add(item)

        # 8. Item-based Collaborative Filtering (ItemCF)
        if self.item_similarity_dict:
            itemcf_cands = []
            for item in list(purchased)[:10]:
                if item in self.item_similarity_dict:
                    for sim_item, score in self.item_similarity_dict[item][:20]:
                        if sim_item not in seen and sim_item not in purchased:
                            itemcf_cands.append((sim_item, score))
                            seen.add(sim_item)
            itemcf_cands.sort(key=lambda x: x[1], reverse=True)
            for item, _ in itemcf_cands[:Config.ITEMCF_CANDIDATES]:
                candidates.append(item)

        # 9. Graph Embedding nearest neighbors
        if self.graph_recaller and self.graph_recaller.article_embedding:
            graph_items = self.graph_recaller.recall(uid, purchased, top_k=Config.GRAPH_CANDIDATES)
            for item in graph_items:
                add(item)

        # 10. Global trending items fallback
        for item in self.hot_items[:Config.HOT_CANDIDATES]:
            add(item)
            if len(candidates) >= top_k:
                break

        # Cold-start enhancement: supplement with additional popular items
        if len(purchased) < 5:
            for item in self.hot_items:
                if item not in seen and item not in purchased:
                    candidates.append(item)
                    seen.add(item)
                    if len(candidates) >= top_k * 1.2:
                        break

        # Final backfill if candidate count is below top_k
        if len(candidates) < top_k:
            for item in self.hot_items:
                if item not in seen and item not in purchased:
                    candidates.append(item)
                    seen.add(item)
                    if len(candidates) >= top_k:
                        break

        return candidates[:top_k], {}, {}

    def recall_batch(self, user_ids, top_k=None, batch_size=5000):
        """Performs batch candidate retrieval across multiple customer IDs."""
        if top_k is None:
            top_k = Config.RECALL_TOP_K
        results = {}
        users = list(user_ids)
        for i in tqdm(range(0, len(users), batch_size), desc="Batch retrieval", leave=False):
            batch = users[i:i + batch_size]
            for uid in batch:
                cands, _, _ = self.recall(uid, top_k)
                results[uid] = np.array(cands, dtype=np.int32)
        return results