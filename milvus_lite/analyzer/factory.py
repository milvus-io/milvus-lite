"""Analyzer factory — builds an Analyzer from analyzer_params dict.

The ``analyzer_params`` dict is the same format that Milvus uses in
``FieldSchema.type_params["analyzer_params"]``.

Supported configurations::

    # String form (simple)
    {"tokenizer": "standard"}
    {"tokenizer": "jieba"}

    # Dict form (with options)
    {"tokenizer": {"type": "jieba", "mode": "search"}}

    # With stop words filter
    {"tokenizer": "standard", "filter": [{"type": "stop", "stop_words": ["the", "a"]}]}
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from milvus_lite.analyzer.protocol import Analyzer
from milvus_lite.analyzer.standard import StandardAnalyzer
from milvus_lite.exceptions import SchemaValidationError


def create_analyzer(params: Optional[Dict[str, Any]] = None) -> Analyzer:
    """Build an Analyzer instance from *params*.

    Args:
        params: analyzer_params dict from FieldSchema, or None for default.

    Returns:
        An Analyzer instance.

    Raises:
        SchemaValidationError: unknown tokenizer type.
        ImportError: jieba requested but not installed.
    """
    if params is None:
        return StandardAnalyzer()

    tokenizer = params.get("tokenizer") or params.get("type", "standard")
    stop_words = _extract_stop_words(params)

    # String form: "standard" / "jieba"
    if isinstance(tokenizer, str):
        return _build_by_name(tokenizer, stop_words=stop_words)

    # Dict form: {"type": "jieba", "mode": "search", ...}
    if isinstance(tokenizer, dict):
        name = tokenizer.get("type", "standard")
        return _build_by_name(
            name,
            stop_words=stop_words,
            mode=tokenizer.get("mode"),
            user_dict_words=tokenizer.get("dict"),
        )

    raise SchemaValidationError(
        f"analyzer_params 'tokenizer' must be a string or dict, "
        f"got {type(tokenizer).__name__}"
    )


def _build_by_name(
    name: str,
    *,
    stop_words: Optional[Set[str]] = None,
    mode: Optional[str] = None,
    user_dict_words: Optional[list] = None,
) -> Analyzer:
    if name == "standard":
        return StandardAnalyzer(stop_words=stop_words)

    if name == "jieba":
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer

        return JiebaAnalyzer(
            mode=mode or "search",
            stop_words=stop_words,
            user_dict_words=user_dict_words,
        )

    raise SchemaValidationError(
        f"unknown tokenizer type: {name!r} "
        f"(supported: 'standard', 'jieba')"
    )


def _extract_stop_words(params: Dict[str, Any]) -> Optional[Set[str]]:
    """Extract stop words from the 'filter' list in params."""
    filters = params.get("filter")
    if not filters or not isinstance(filters, list):
        return None

    words: set[str] = set()
    for f in filters:
        if isinstance(f, dict) and f.get("type") == "stop":
            sw = f.get("stop_words", [])
            if isinstance(sw, (list, set)):
                words.update(str(w) for w in sw)
    return words if words else None
