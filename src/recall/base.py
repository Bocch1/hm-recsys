"""
Base retrieval module defining the unified interface for all recall strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseRecall(ABC):
    """Abstract base class for all candidate retrieval strategies."""

    def __init__(self, name: str):
        """
        Initializes the recall instance.

        Args:
            name (str): Identifier used for logging, profiling, and registry.
        """
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs):
        """
        Fits or aggregates historical data required for candidate generation.

        Args:
            train_df (pd.DataFrame): Training transactions containing 'customer_id' and 'article_id'.
            **kwargs: Additional strategy-specific arguments.
        """
        pass

    @abstractmethod
    def recall(self, user_ids: List[str], top_k: int = 100) -> Dict[str, List[str]]:
        """
        Generates candidate items for a given list of customer identifiers.

        Args:
            user_ids (List[str]): List of target customer identifiers.
            top_k (int): Number of candidate items to retrieve per customer.

        Returns:
            Dict[str, List[str]]: Mapping from customer_id to a list of candidate article_ids,
                                  sorted in descending order of relevance score.
        """
        pass

    def get_name(self) -> str:
        """Returns the name identifier of the recall strategy."""
        return self.name

    def is_ready(self) -> bool:
        """Checks whether the recaller has been fitted."""
        return self.is_fitted

    def _validate_fitted(self):
        """Validates that the recaller is fitted before generating candidates."""
        if not self.is_fitted:
            raise RuntimeError(f"Recall strategy '{self.name}' has not been fitted. Call fit() first.")