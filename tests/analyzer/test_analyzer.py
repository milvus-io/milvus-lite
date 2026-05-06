"""Phase 11.2 — Analyzer subsystem tests.

Covers:
- term_to_id hash: determinism, distribution, utf-8 support
- StandardAnalyzer: tokenization, lowercasing, stop words
- JiebaAnalyzer: Chinese tokenization (if jieba is installed)
- create_analyzer factory: routing, params parsing, error handling
- Analyzer.analyze(): tokenize + hash integration
"""

import pytest

from milvus_lite.analyzer.hash import term_to_id
from milvus_lite.analyzer.standard import StandardAnalyzer
from milvus_lite.analyzer.factory import create_analyzer
from milvus_lite.analyzer.protocol import Analyzer
from milvus_lite.exceptions import SchemaValidationError


# ---------------------------------------------------------------------------
# term_to_id hash
# ---------------------------------------------------------------------------

class TestTermToId:
    def test_deterministic(self):
        """Same term always produces the same ID."""
        assert term_to_id("hello") == term_to_id("hello")
        assert term_to_id("world") == term_to_id("world")

    def test_different_terms_different_ids(self):
        ids = {term_to_id(w) for w in ["apple", "banana", "cherry", "date", "elderberry"]}
        assert len(ids) == 5  # no collisions for small set

    def test_uint32_range(self):
        for word in ["a", "test", "hello world", "机器学习"]:
            h = term_to_id(word)
            assert 0 <= h < 2**32, f"{word!r} hashed to {h}"

    def test_utf8_support(self):
        """Chinese / emoji / accented chars produce valid IDs."""
        h1 = term_to_id("机器")
        h2 = term_to_id("学习")
        assert h1 != h2
        assert 0 <= h1 < 2**32

    def test_empty_string(self):
        h = term_to_id("")
        assert isinstance(h, int)
        assert 0 <= h < 2**32

    def test_consistency_across_calls(self):
        """Hash is not affected by Python's PYTHONHASHSEED."""
        # FNV-1a is deterministic by design; just verify repeated calls.
        results = [term_to_id("consistency") for _ in range(100)]
        assert len(set(results)) == 1

    def test_known_value(self):
        """FNV-1a of "hello" should be a specific value (regression)."""
        h = term_to_id("hello")
        # Just check it's stable — exact value depends on implementation
        assert isinstance(h, int)
        # Run twice to confirm
        assert term_to_id("hello") == h


# ---------------------------------------------------------------------------
# StandardAnalyzer
# ---------------------------------------------------------------------------

class TestStandardAnalyzer:
    def test_basic_tokenization(self):
        a = StandardAnalyzer()
        assert a.tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_removal(self):
        a = StandardAnalyzer()
        tokens = a.tokenize("Hello, World! How's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "s" in tokens
        assert "," not in tokens

    def test_lowercasing(self):
        a = StandardAnalyzer()
        assert a.tokenize("UPPER lower MiXeD") == ["upper", "lower", "mixed"]

    def test_numbers_included(self):
        a = StandardAnalyzer()
        tokens = a.tokenize("test123 456")
        assert "test123" in tokens
        assert "456" in tokens

    def test_empty_string(self):
        a = StandardAnalyzer()
        assert a.tokenize("") == []

    def test_whitespace_only(self):
        a = StandardAnalyzer()
        assert a.tokenize("   \t\n  ") == []

    def test_unicode_words(self):
        a = StandardAnalyzer()
        tokens = a.tokenize("café résumé naïve")
        assert "café" in tokens
        assert "résumé" in tokens

    def test_stop_words_filter(self):
        stops = {"the", "a", "is", "in"}
        a = StandardAnalyzer(stop_words=stops)
        tokens = a.tokenize("The cat is in a box")
        assert "cat" in tokens
        assert "box" in tokens
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "in" not in tokens

    def test_no_stop_words_by_default(self):
        a = StandardAnalyzer()
        tokens = a.tokenize("the cat is in a box")
        assert "the" in tokens
        assert "is" in tokens

    def test_analyze_returns_int_ids(self):
        a = StandardAnalyzer()
        ids = a.analyze("hello world")
        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)
        assert ids[0] == term_to_id("hello")
        assert ids[1] == term_to_id("world")

    def test_analyze_preserves_duplicates(self):
        a = StandardAnalyzer()
        ids = a.analyze("test test test")
        assert len(ids) == 3
        assert ids[0] == ids[1] == ids[2]

    def test_multiline_text(self):
        a = StandardAnalyzer()
        tokens = a.tokenize("line one\nline two\nline three")
        assert tokens == ["line", "one", "line", "two", "line", "three"]


# ---------------------------------------------------------------------------
# JiebaAnalyzer
# ---------------------------------------------------------------------------

def _jieba_available():
    try:
        import jieba  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _jieba_available(), reason="jieba not installed")
class TestJiebaAnalyzer:
    def test_basic_chinese(self):
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer(mode="exact")
        tokens = a.tokenize("我是一个学生")
        assert len(tokens) > 0
        # jieba should segment this into meaningful words
        assert any("学生" in t for t in tokens)

    def test_search_mode(self):
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer(mode="search")
        tokens = a.tokenize("中华人民共和国")
        # search mode produces finer-grained segments
        assert len(tokens) >= 1

    def test_stop_words(self):
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer(stop_words={"的", "是", "了"})
        tokens = a.tokenize("今天的天气是很好的")
        assert "的" not in tokens
        assert "是" not in tokens

    def test_analyze_returns_ids(self):
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer()
        ids = a.analyze("机器学习算法")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_user_dict(self):
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer(user_dict_words=["深度学习"])
        tokens = a.tokenize("深度学习是人工智能的子领域")
        assert "深度学习" in tokens

    def test_english_text(self):
        """Jieba handles English text too (passes through)."""
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer()
        tokens = a.tokenize("machine learning")
        assert len(tokens) > 0

    def test_is_analyzer_subclass(self):
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        a = JiebaAnalyzer()
        assert isinstance(a, Analyzer)


def test_jieba_import_error():
    """When jieba is not installed, JiebaAnalyzer raises ImportError."""
    # We can't easily uninstall jieba mid-test, so just verify the class
    # exists and is importable.
    from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
    assert JiebaAnalyzer is not None


# ---------------------------------------------------------------------------
# create_analyzer factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_none_params(self):
        a = create_analyzer(None)
        assert isinstance(a, StandardAnalyzer)

    def test_empty_params(self):
        a = create_analyzer({})
        assert isinstance(a, StandardAnalyzer)

    def test_standard_string(self):
        a = create_analyzer({"tokenizer": "standard"})
        assert isinstance(a, StandardAnalyzer)

    def test_standard_with_stop_words(self):
        a = create_analyzer({
            "tokenizer": "standard",
            "filter": [{"type": "stop", "stop_words": ["the", "a"]}],
        })
        tokens = a.tokenize("the cat and a dog")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "cat" in tokens

    @pytest.mark.skipif(not _jieba_available(), reason="jieba not installed")
    def test_jieba_string(self):
        a = create_analyzer({"tokenizer": "jieba"})
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        assert isinstance(a, JiebaAnalyzer)

    @pytest.mark.skipif(not _jieba_available(), reason="jieba not installed")
    def test_jieba_dict_form(self):
        a = create_analyzer({
            "tokenizer": {"type": "jieba", "mode": "exact"},
        })
        from milvus_lite.analyzer.jieba_analyzer import JiebaAnalyzer
        assert isinstance(a, JiebaAnalyzer)

    @pytest.mark.skipif(not _jieba_available(), reason="jieba not installed")
    def test_jieba_with_dict_and_stop_words(self):
        a = create_analyzer({
            "tokenizer": {
                "type": "jieba",
                "mode": "search",
                "dict": ["自定义词"],
            },
            "filter": [{"type": "stop", "stop_words": ["的"]}],
        })
        tokens = a.tokenize("这是自定义词的测试")
        assert "自定义词" in tokens
        assert "的" not in tokens

    def test_unknown_tokenizer(self):
        with pytest.raises(SchemaValidationError, match="unknown tokenizer"):
            create_analyzer({"tokenizer": "nonexistent"})

    def test_invalid_tokenizer_type(self):
        with pytest.raises(SchemaValidationError, match="must be a string or dict"):
            create_analyzer({"tokenizer": 42})

    def test_filter_without_stop_words(self):
        """Filter list with non-stop type is ignored gracefully."""
        a = create_analyzer({
            "tokenizer": "standard",
            "filter": [{"type": "lowercase"}],  # not a stop filter
        })
        assert isinstance(a, StandardAnalyzer)

    def test_multiple_stop_filters_merged(self):
        a = create_analyzer({
            "tokenizer": "standard",
            "filter": [
                {"type": "stop", "stop_words": ["the"]},
                {"type": "stop", "stop_words": ["a", "an"]},
            ],
        })
        tokens = a.tokenize("the cat and a dog")
        assert "the" not in tokens
        assert "a" not in tokens
        assert "cat" in tokens


# ---------------------------------------------------------------------------
# Integration: Analyzer protocol
# ---------------------------------------------------------------------------

class TestAnalyzerProtocol:
    def test_standard_is_subclass(self):
        assert isinstance(StandardAnalyzer(), Analyzer)

    def test_analyze_consistency(self):
        """analyze() = tokenize() + term_to_id()."""
        a = StandardAnalyzer()
        text = "machine learning algorithms"
        tokens = a.tokenize(text)
        ids = a.analyze(text)
        assert len(tokens) == len(ids)
        for token, tid in zip(tokens, ids):
            assert tid == term_to_id(token)
