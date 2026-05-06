"""StandardAnalyzer — regex-based tokenizer for Latin-script text.

Splits on non-word characters, lowercases, and optionally filters
stop words.  Zero external dependencies.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from milvus_lite.analyzer.protocol import Analyzer


# A small default English stop-word set (can be overridden via params).
_DEFAULT_STOP_WORDS: frozenset[str] = frozenset()


class StandardAnalyzer(Analyzer):
    """Regex tokenizer: ``\\w+`` + lowercase + optional stop words."""

    def __init__(self, stop_words: Optional[Set[str]] = None) -> None:
        self._stop_words: frozenset[str] = (
            frozenset(stop_words) if stop_words else _DEFAULT_STOP_WORDS
        )
        self._pattern = re.compile(r"\w+", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        tokens = self._pattern.findall(text.lower())
        if self._stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]
        return tokens
