"""JiebaAnalyzer — Chinese text tokenizer (optional dependency).

Requires ``jieba`` to be installed (``pip install jieba``).  If jieba
is not available, ``create_analyzer`` raises a clear error at schema
creation time — it never silently falls back.
"""

from __future__ import annotations

from typing import List, Optional, Set

from milvus_lite.analyzer.protocol import Analyzer


class JiebaAnalyzer(Analyzer):
    """Chinese tokenizer backed by the ``jieba`` library.

    Args:
        mode: ``"search"`` (cut_for_search, finer granularity) or
              ``"exact"`` (cut with HMM, standard segmentation).
        stop_words: optional set of words to filter out.
        user_dict_words: optional list of custom dictionary entries
            (each entry is a Chinese word string, e.g. ``"deep learning"``).
    """

    def __init__(
        self,
        mode: str = "search",
        stop_words: Optional[Set[str]] = None,
        user_dict_words: Optional[List[str]] = None,
    ) -> None:
        try:
            import jieba as _jieba  # noqa: F811
        except ImportError as e:
            raise ImportError(
                "JiebaAnalyzer requires the 'jieba' package. "
                "Install it with: pip install jieba"
            ) from e

        self._jieba = _jieba
        self._mode = mode
        self._stop_words: frozenset[str] = (
            frozenset(stop_words) if stop_words else frozenset()
        )

        if user_dict_words:
            for word in user_dict_words:
                self._jieba.add_word(word)

    def tokenize(self, text: str) -> List[str]:
        if self._mode == "search":
            raw = self._jieba.cut_for_search(text)
        else:
            raw = self._jieba.cut(text)

        tokens = [t.strip().lower() for t in raw if t.strip()]
        if self._stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]
        return tokens
