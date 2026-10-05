"""
Feature engineering and multimodal representation package.
Exports tabular feature construction pipelines and multimodal embedding extractors.
"""

from .build_features import (
    build_interaction_features,
    build_item_features,
    build_item_stats,
    build_mappings,
    build_time_window_features,
    build_user_features,
    build_user_stats,
    run_all,
)
from .image_text_embedding import ImageTextEmbeddingExtractor

__all__ = [
    'build_item_features',
    'build_user_features',
    'build_time_window_features',
    'build_interaction_features',
    'build_user_stats',
    'build_item_stats',
    'build_mappings',
    'run_all',
    'ImageTextEmbeddingExtractor',
]