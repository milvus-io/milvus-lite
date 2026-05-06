"""EmbeddingProvider protocol — abstract embedding model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding models.

    Implementations must handle batching and API rate limits internally.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts.

        Args:
            texts: list of document strings

        Returns:
            list of embedding vectors, same length as texts
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Some providers distinguish document vs query embeddings
        (e.g., asymmetric models). Default: same as embed_documents
        with a single input.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""
