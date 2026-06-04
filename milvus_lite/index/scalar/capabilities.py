from __future__ import annotations

from milvus_lite.schema.types import DataType


IMPLEMENTED_SCALAR_INDEX_TYPES = frozenset({"INVERTED"})
KNOWN_SCALAR_INDEX_TYPES = frozenset({"INVERTED", "BITMAP", "STL_SORT", "TRIE", "NGRAM"})
SUPPORTED_SCALAR_INDEX_DTYPES = frozenset({
    DataType.BOOL,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.VARCHAR,
    DataType.TIMESTAMPTZ,
})
SUPPORTED_DTYPES_BY_SCALAR_INDEX_TYPE = {
    "INVERTED": SUPPORTED_SCALAR_INDEX_DTYPES,
}
