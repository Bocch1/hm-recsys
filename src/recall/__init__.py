"""
Candidate retrieval package.
Exports base retrieval interfaces, multi-channel recall strategies, and ensemble mergers.
"""

from .base import BaseRecall
from .hot_recall import HotRecall
from .itemcf_recall import ItemCFRecall
from .embedding_recall import EmbeddingRecall
from .image_recall import ImageRecall
from .award_recall import AwardRecall
from .merger import RecallMerger

__all__ = [
    'BaseRecall',
    'HotRecall',
    'ItemCFRecall',
    'EmbeddingRecall',
    'ImageRecall',
    'AwardRecall',
    'RecallMerger',
]