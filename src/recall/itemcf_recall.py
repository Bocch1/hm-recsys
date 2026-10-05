"""
Item-based Collaborative Filtering (ItemCF) candidate retrieval module.
"""

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecall


class ItemCFRecall(BaseRecall):
    """
    Item-based Collaborative Filtering candidate retrieval engine.

    Computes item-to-item cosine similarity over user interaction vectors
    and generates candidate recommendations from historical customer interactions.

    Parameters
    ----------
    top_k : int, default=100
        Number of items to retrieve per customer.
    hot_item_threshold : int, default=5000
        Number of top popular items included in similarity matrix computation.
    min_sim : float, default=0.01
        Minimum similarity threshold required to preserve item connections.
    name : str, default='itemcf'
        Identifier name of the recall strategy.
    """

    def __init__(
        self,
        top_k: int = 100,
        hot_item_threshold: int = 5000,
        min_sim: float = 0.01,
        name: str = 'itemcf'
    ):
        super().__init__(name)
        self.top_k = top_k
        self.hot_item_threshold = hot_item_threshold
        self.min_sim = min_sim

        self.item_similarity = None
        self.hot_items = None
        self.hot_item_to_idx = None
        self.idx_to_hot_item = None
        self.user_history = None
        self._all_items = None

    def fit(self, train_df: pd.DataFrame, **kwargs):
        """
        Builds sparse user-item interaction matrix and computes item similarity indices.
        """
        if 'customer_id' not in train_df.columns or 'article_id' not in train_df.columns:
            raise ValueError("Training dataframe must contain 'customer_id' and 'article_id' columns.")

        print(f"[{self.name}] Fitting ItemCF model...")

        item_counts = train_df['article_id'].value_counts()
        self._all_items = item_counts.index.tolist()

        hot_items_full = item_counts.head(self.hot_item_threshold).index.tolist()
        print(f"  - Active items for similarity index: {len(hot_items_full):,}")

        train_hot = train_df[train_df['article_id'].isin(hot_items_full)]

        self.hot_item_to_idx = {item: i for i, item in enumerate(hot_items_full)}
        self.idx_to_hot_item = {i: item for i, item in enumerate(hot_items_full)}
        self.hot_items = hot_items_full

        user_ids = train_hot['customer_id'].unique()
        user_to_idx = {u: i for i, u in enumerate(user_ids)}

        rows = [user_to_idx[u] for u in train_hot['customer_id']]
        cols = [self.hot_item_to_idx[item] for item in train_hot['article_id']]

        temp_df = pd.DataFrame({'user': rows, 'item': cols})
        agg = temp_df.groupby(['user', 'item']).size().reset_index(name='count')

        user_item_matrix = csr_matrix(
            (agg['count'].values, (agg['user'].values, agg['item'].values)),
            shape=(len(user_ids), len(hot_items_full))
        )

        print(f"  - Interaction matrix shape: {user_item_matrix.shape}")
        print(f"  - Non-zero interactions:    {user_item_matrix.nnz:,}")

        print("  - Computing item-item cosine similarity...")
        similarity = cosine_similarity(user_item_matrix.T, dense_output=False)

        print("  - Filtering low-similarity edges and indexing neighbors...")
        self.item_similarity = defaultdict(dict)

        for i in range(similarity.shape[0]):
            row = similarity[i]
            if hasattr(row, 'nonzero'):
                indices = row.nonzero()[1]
                values = row.data
            else:
                indices = np.where(row > self.min_sim)[0]
                values = row[indices]

            if len(indices) > 0:
                sorted_idx = np.argsort(values)[::-1]
                top_indices = []
                top_values = []
                count = 0
                for idx in sorted_idx:
                    if idx != i and count < 50:
                        top_indices.append(idx)
                        top_values.append(values[idx])
                        count += 1

                hot_item = self.idx_to_hot_item[i]
                for j, sim in zip(top_indices, top_values):
                    if sim >= self.min_sim:
                        similar_item = self.idx_to_hot_item[j]
                        self.item_similarity[hot_item][similar_item] = float(sim)

        self.user_history = {}
        for user_id, group in train_df.groupby('customer_id'):
            self.user_history[user_id] = group['article_id'].tolist()

        self.is_fitted = True

        total_sim_items = sum(len(v) for v in self.item_similarity.values())
        print(f"[{self.name}] Model fitting completed.")
        print(f"  - Indexed items:            {len(self.hot_items):,}")
        print(f"  - Total similarity edges:   {total_sim_items:,}")
        print(f"  - Mean neighbors per item:  {total_sim_items / max(len(self.hot_items), 1):.2f}")

        return self

    def recall(self, user_ids: List[str], top_k: Optional[int] = None) -> Dict[str, List[str]]:
        self._validate_fitted()
        if top_k is None:
            top_k = self.top_k

        recommendations = {}

        for user_id in user_ids:
            purchased = set(self.user_history.get(user_id, []))
            candidate_scores = defaultdict(float)

            for item in purchased:
                if item in self.item_similarity:
                    for similar_item, sim_score in self.item_similarity[item].items():
                        if similar_item not in purchased:
                            candidate_scores[similar_item] += sim_score

            if len(candidate_scores) < top_k:
                hot_items = self.hot_items[:top_k]
                for item in hot_items:
                    if item not in purchased and item not in candidate_scores:
                        candidate_scores[item] += 0.01

            sorted_candidates = sorted(
                candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            recommendations[user_id] = [item for item, _ in sorted_candidates[:top_k]]

        return recommendations