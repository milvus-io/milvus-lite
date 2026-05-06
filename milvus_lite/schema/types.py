"""Data type definitions for MilvusLite schema layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

import pyarrow as pa


class DataType(Enum):
    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    DOUBLE = "double"
    VARCHAR = "varchar"
    JSON = "json"
    ARRAY = "array"
    FLOAT_VECTOR = "float_vector"
    SPARSE_FLOAT_VECTOR = "sparse_float_vector"


class FunctionType(IntEnum):
    """Function types that can be attached to a CollectionSchema."""
    BM25 = 1
    TEXT_EMBEDDING = 2
    RERANK = 3


@dataclass
class Function:
    """A schema-level function that auto-generates output fields from input fields.

    For BM25: input is a VARCHAR field (with enable_analyzer=True),
    output is a SPARSE_FLOAT_VECTOR field.
    """
    name: str
    function_type: FunctionType
    input_field_names: List[str]
    output_field_names: List[str]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldSchema:
    name: str
    dtype: DataType
    is_primary: bool = False
    auto_id: bool = False
    dim: Optional[int] = None
    max_length: Optional[int] = None
    element_type: Optional[DataType] = None  # For ARRAY fields
    max_capacity: Optional[int] = None       # For ARRAY fields
    nullable: bool = False
    default_value: Any = None
    # Full text search attributes (Phase 11)
    enable_analyzer: bool = False
    analyzer_params: Optional[Dict[str, Any]] = None
    enable_match: bool = False
    is_function_output: bool = False
    is_partition_key: bool = False


@dataclass
class CollectionSchema:
    fields: List[FieldSchema]
    version: int = 1
    enable_dynamic_field: bool = False
    functions: List[Function] = field(default_factory=list)


# DataType -> PyArrow type mapping.
# FLOAT_VECTOR is None here; at runtime use pa.list_(pa.float32(), dim).
# SPARSE_FLOAT_VECTOR uses pa.binary() (packed uint32+float32 pairs).
TYPE_MAP: Dict[DataType, Any] = {
    DataType.BOOL: pa.bool_(),
    DataType.INT8: pa.int8(),
    DataType.INT16: pa.int16(),
    DataType.INT32: pa.int32(),
    DataType.INT64: pa.int64(),
    DataType.FLOAT: pa.float32(),
    DataType.DOUBLE: pa.float64(),
    DataType.VARCHAR: pa.string(),
    DataType.JSON: pa.string(),
    DataType.ARRAY: None,  # resolved at runtime from element_type
    DataType.FLOAT_VECTOR: None,
    DataType.SPARSE_FLOAT_VECTOR: pa.binary(),
}
