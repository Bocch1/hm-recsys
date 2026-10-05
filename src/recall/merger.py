"""
Multi-channel candidate aggregation and rank fusion module.
"""

from collections import defaultdict
from typing import Dict, List, Optional


class RecallMerger:
    """
    Multi-channel candidate merger.

    Features:
    - Weighted rank-based fusion across disparate retrieval channels.
    - Reciprocal rank weighting: score = weight / (rank + 1).
    - Deduplication and truncation to target candidate pool depth.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights (dict, optional): Mapping from recall strategy names to fusion weights,
                                      e.g., {'hot_recall': 0.2, 'itemcf': 0.8}.
                                      Defaults to uniform weighting if not provided.
        """
        self.weights = weights or {}
        self.recallers = []

    def add_recaller(self, name: str, weight: float = 1.0):
        """Registers a retrieval channel along with its fusion weight."""
        self.weights[name] = weight
        return self

    def merge(
        self,
        all_results: Dict[str, Dict[str, List[str]]],
        top_k: int = 200
    ) -> Dict[str, List[str]]:
        """
        Merges multi-channel candidate lists using rank-based score aggregation.

        Args:
            all_results: Mapping of format {recaller_name: {user_id: [article_id_list]}}.
            top_k: Number of candidates to retain per customer.

        Returns:
            Dict[str, List[str]]: Mapping from user_id to deduplicated candidate lists
                                  sorted in descending order of weighted score.
        """
        if not all_results:
            return {}

        all_users = set()
        for results in all_results.values():
            all_users.update(results.keys())

        merged = {}

        for user_id in all_users:
            scores = defaultdict(float)

            for recaller_name, results in all_results.items():
                if user_id not in results:
                    continue

                weight = self.weights.get(recaller_name, 1.0)
                rec_list = results[user_id]

                for rank, item in enumerate(rec_list):
                    rank_score = 1.0 / (rank + 1)
                    scores[item] += weight * rank_score

            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            merged[user_id] = [item for item, _ in sorted_items[:top_k]]

        return merged

    def merge_simple(
        self,
        all_results: Dict[str, Dict[str, List[str]]],
        top_k: int = 200
    ) -> Dict[str, List[str]]:
        """
        Interleaved round-robin candidate merging with deduplication.
        """
        if not all_results:
            return {}

        all_users = set()
        for results in all_results.values():
            all_users.update(results.keys())

        merged = {}

        for user_id in all_users:
            user_items = []
            for results in all_results.values():
                if user_id in results:
                    user_items.append(results[user_id])

            combined = []
            seen = set()
            max_len = max(len(items) for items in user_items) if user_items else 0

            for i in range(max_len):
                for items in user_items:
                    if i < len(items) and items[i] not in seen:
                        combined.append(items[i])
                        seen.add(items[i])
                        if len(combined) >= top_k:
                            break
                if len(combined) >= top_k:
                    break

            merged[user_id] = combined[:top_k]

        return merged