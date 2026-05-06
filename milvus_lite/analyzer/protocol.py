"""Analyzer abstract base class for text tokenization.

An Analyzer converts raw text into a list of tokens (strings) or
term IDs (integers). The engine uses analyzers at two points:

1. **Insert time**: tokenize text → compute term frequencies → store
   as SPARSE_FLOAT_VECTOR.
2. **Search time**: tokenize query text → look up inverted index.

Implementations: StandardAnalyzer (regex), JiebaAnalyzer (Chinese).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from milvus_lite.analyzer.hash import term_to_id


class Analyzer(ABC):
    """Base class for text analyzers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split *text* into a list of token strings.

        Implementations should handle lowercasing, punctuation removal,
        stop-word filtering, etc.
        """

    def analyze(self, text: str) -> List[int]:
        """Tokenize *text* and map each token to a uint32 term ID.

        Returns a list of term IDs (may contain duplicates — the caller
        counts frequencies).
        """
        return [term_to_id(t) for t in self.tokenize(text)]
