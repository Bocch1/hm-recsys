"""
Global popularity-based candidate retrieval module.
"""

from typing import Dict, List, Optional
import pandas as pd
from .base import BaseRecall


class HotRecall(BaseRecall):
    """
    Global popularity recall engine.

    Returns the top globally trending items across the entire customer base.
    Serves as the fundamental cold-start fallback and baseline benchmark.

    Parameters
    ----------
    top_k : int, default=100
        Number of items to retrieve per customer.
    name : str, default='hot_recall'
        Identifier name of the recall strategy.
    """

    def __init__(self, top_k: int = 100, name: str = 'hot_recall'):
        super().__init__(name)
        self.top_k = top_k
        self.hot_items = None
        self.hot_items_scores = None
        self._all_items = None

    def fit(self, train_df: pd.DataFrame, **kwargs):
        """
        Aggregates transaction frequencies to identify top trending items.
        """
        if 'article_id' not in train_df.columns:
            raise ValueError("Training dataframe must contain 'article_id' column.")

        item_counts = train_df['article_id'].value_counts()
        self._all_items = item_counts.index.tolist()

        self.hot_items = item_counts.head(self.top_k).index.tolist()
        self.hot_items_scores = item_counts.head(self.top_k).values.tolist()

        self.is_fitted = True

        print(f"[{self.name}] Fitting completed successfully.")
        print(f"  - Total catalog items: {len(self._all_items):,}")
        print(f"  - Retrieved top-k:     {self.top_k}")
        if self.hot_items:
            print(f"  - Top-1 item:         {self.hot_items[0]} ({self.hot_items_scores[0]:,} purchases)")
            print(f"  - Top-3 items:        {self.hot_items[:3]}")

        return self

    def recall(self, user_ids: List[str], top_k: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Broadcasts top global trending items to all requested customer IDs.
        """
        self._validate_fitted()

        if top_k is None:
            top_k = self.top_k

        if len(self.hot_items) < top_k:
            additional = [item for item in self._all_items if item not in self.hot_items]
            full_list = self.hot_items + additional[:top_k - len(self.hot_items)]
        else:
            full_list = self.hot_items[:top_k]

        recommendations = {user_id: full_list.copy() for user_id in user_ids}
        return recommendations

    def get_top_n(self, n: Optional[int] = None) -> List[str]:
        """Returns the top N globally popular items."""
        self._validate_fitted()
        if n is None:
            n = self.top_k
        return self.hot_items[:n]