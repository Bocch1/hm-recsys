"""
Evaluation metrics module for recommendation systems.
Implements Mean Average Precision at 12 (MAP@12), Recall@K, and HitRate@K.
"""

from typing import Dict, List, Optional
import numpy as np


def compute_map12(
    recommendations: Dict[str, List[str]], 
    test_truth_dict: Dict[str, List[str]]
) -> float:
    """
    Compute Mean Average Precision at 12 (MAP@12).
    Official competition evaluation metric for H&M Personalized Fashion Recommendations.

    Formula:
        MAP@12 = (1 / U) * sum_u [ (1 / min(m, 12)) * sum_{k=1}^{min(n, 12)} P(k) * rel(k) ]

    Args:
        recommendations: Dictionary mapping customer_id to a list of predicted article_ids.
        test_truth_dict: Dictionary mapping customer_id to a list of ground truth article_ids.

    Returns:
        float: Mean Average Precision at 12 across all evaluated customers.
    """
    average_precisions = []

    for uid, true_items in test_truth_dict.items():
        if uid not in recommendations:
            average_precisions.append(0.0)
            continue

        rec_list = recommendations[uid][:12]
        true_set = set(true_items)

        m = len(true_items)
        n = len(rec_list)

        if m == 0 or n == 0:
            average_precisions.append(0.0)
            continue

        norm_factor = 1.0 / min(m, 12)
        hits = 0
        precision_sum = 0.0

        for k in range(1, min(n, 12) + 1):
            item = rec_list[k - 1]
            if item in true_set:
                hits += 1
                p_at_k = hits / k
                precision_sum += p_at_k

        ap = norm_factor * precision_sum
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def evaluate_recall_multi(
    recommendations: Dict[str, List[str]], 
    test_truth_dict: Dict[str, List[str]], 
    k_values: Optional[List[int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute Recall@K and HitRate@K across multiple cutoff thresholds.

    Args:
        recommendations: Dictionary mapping customer_id to a list of predicted article_ids.
        test_truth_dict: Dictionary mapping customer_id to a list of ground truth article_ids.
        k_values: List of cutoff depths to evaluate. Defaults to [10, 20, 50].

    Returns:
        Dict[int, Dict[str, float]]: Metric results containing 'recall' and 'hit_rate' for each K.
    """
    if k_values is None:
        k_values = [10, 20, 50]

    results = {}
    for k in k_values:
        total_hits = 0
        total_truth = 0
        user_hit_count = 0

        for uid, true_items in test_truth_dict.items():
            if uid not in recommendations:
                continue

            rec_list = recommendations[uid][:k]
            true_set = set(true_items)

            hits = sum(1 for item in rec_list if item in true_set)
            total_hits += hits
            total_truth += len(true_set)

            if hits > 0:
                user_hit_count += 1

        results[k] = {
            'recall': total_hits / total_truth if total_truth > 0 else 0.0,
            'hit_rate': user_hit_count / len(test_truth_dict) if test_truth_dict else 0.0,
        }

    return results