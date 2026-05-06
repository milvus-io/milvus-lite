"""RerankProvider protocol — abstract reranking model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RerankResult:
    """A single reranked document result.

    Attributes:
        index: position in the original document list
        relevance_score: reranker relevance score (higher = more relevant)
    """
    index: int
    relevance_score: float


class RerankProvider(ABC):
    """Abstract interface for reranking models.

    Implementations call an external cross-encoder / reranker API to
    re-score documents by relevance to a query.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to a query.

        Args:
            query: the search query text
            documents: list of document texts to rerank
            top_n: if set, return only the top N results.
                   If None, return all documents reranked.

        Returns:
            list of RerankResult sorted by relevance_score descending
        """
