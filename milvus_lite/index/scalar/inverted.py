"""Segment-level scalar INVERTED index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from milvus_lite.index.scalar.capabilities import SUPPORTED_SCALAR_INDEX_DTYPES
from milvus_lite.index.scalar.planner import ScalarPredicate
from milvus_lite.schema.timestamptz import parse_timestamptz
from milvus_lite.schema.types import DataType


_SCALAR_INDEX_VERSION = "1"


@dataclass
class ScalarInvertedIndex:
    field_name: str
    dtype: DataType
    row_count: int
    null_mask: np.ndarray
    value_to_rows: Dict[Any, np.ndarray]
    sorted_values: np.ndarray
    sorted_rows: np.ndarray

    @classmethod
    def build(cls, table: pa.Table, field_name: str, dtype: DataType) -> "ScalarInvertedIndex":
        if dtype not in SUPPORTED_SCALAR_INDEX_DTYPES:
            raise NotImplementedError(
                f"scalar index INVERTED does not support field type {dtype.name}"
            )
        normalized = [
            _normalize_value(value, dtype)
            for value in table.column(field_name).to_pylist()
        ]
        return cls._from_normalized_values(field_name, dtype, normalized)

    @classmethod
    def load(cls, path: str) -> "ScalarInvertedIndex":
        with pa.memory_map(path, "r") as source:
            reader = ipc.RecordBatchFileReader(source)
            metadata = reader.schema.metadata or {}
            if metadata.get(b"milvus_lite.scalar_index.version") != _SCALAR_INDEX_VERSION.encode():
                raise ValueError(f"unsupported scalar index file version at {path!r}")
            field_name = metadata[b"field_name"].decode()
            dtype = DataType[metadata[b"dtype"].decode()]
            row_count = int(metadata[b"row_count"].decode())
            table = reader.read_all()

        values = table.column("value").to_pylist()
        row_ids = table.column("row_id").to_numpy(zero_copy_only=False)
        is_null = table.column("is_null").to_numpy(zero_copy_only=False)

        normalized: List[Any] = [None] * row_count
        for value, row_id, null_flag in zip(values, row_ids, is_null):
            if not null_flag:
                normalized[int(row_id)] = _normalize_value(value, dtype)

        return cls._from_normalized_values(field_name, dtype, normalized)

    @classmethod
    def _from_normalized_values(
        cls,
        field_name: str,
        dtype: DataType,
        normalized: List[Any],
    ) -> "ScalarInvertedIndex":
        row_count = len(normalized)
        null_mask = np.asarray([value is None for value in normalized], dtype=bool)
        buckets: Dict[Any, List[int]] = {}
        sorted_pairs: List[tuple[Any, int]] = []
        for row_id, value in enumerate(normalized):
            if value is None:
                continue
            buckets.setdefault(value, []).append(row_id)
            sorted_pairs.append((value, row_id))

        value_to_rows = {
            value: np.asarray(rows, dtype=np.int64)
            for value, rows in buckets.items()
        }
        sorted_pairs.sort(key=lambda item: item[0])
        sorted_values = np.asarray([p[0] for p in sorted_pairs], dtype=_numpy_dtype(dtype))
        sorted_rows = np.asarray([p[1] for p in sorted_pairs], dtype=np.int64)
        return cls(field_name, dtype, row_count, null_mask, value_to_rows, sorted_values, sorted_rows)

    def save(self, path: str) -> None:
        null_rows = np.flatnonzero(self.null_mask).astype(np.int64, copy=False)
        values = self.sorted_values.tolist() + [None] * int(null_rows.size)
        row_ids = self.sorted_rows.astype(np.int64, copy=False).tolist() + null_rows.tolist()
        is_null = [False] * int(self.sorted_rows.size) + [True] * int(null_rows.size)

        table = pa.Table.from_arrays(
            [
                pa.array(values, type=_arrow_type(self.dtype)),
                pa.array(row_ids, type=pa.int64()),
                pa.array(is_null, type=pa.bool_()),
            ],
            names=["value", "row_id", "is_null"],
        )
        metadata = {
            b"milvus_lite.scalar_index.version": _SCALAR_INDEX_VERSION.encode(),
            b"field_name": self.field_name.encode(),
            b"dtype": self.dtype.name.encode(),
            b"row_count": str(self.row_count).encode(),
        }
        table = table.replace_schema_metadata(metadata)
        with pa.OSFile(path, "wb") as sink:
            with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

    def match(self, predicate: ScalarPredicate) -> np.ndarray:
        op = predicate.op
        if op == "==":
            return self._eq(predicate.value)
        if op == "!=":
            return (~self._eq(predicate.value)) & (~self.null_mask)
        if op == "in":
            return self._in(predicate.values)
        if op == "not in":
            return ~self._in(predicate.values)
        if op in ("<", "<=", ">", ">="):
            return self._range(op, predicate.value)
        if op == "is null":
            return self.null_mask.copy()
        if op == "is not null":
            return ~self.null_mask
        raise ValueError(f"unsupported scalar predicate op: {op!r}")

    def _eq(self, value: Any) -> np.ndarray:
        mask = np.zeros(self.row_count, dtype=bool)
        normalized = _normalize_value(value, self.dtype)
        if normalized is None:
            return mask
        rows = self.value_to_rows.get(normalized)
        if rows is not None:
            mask[rows] = True
        return mask

    def _in(self, values: tuple[Any, ...]) -> np.ndarray:
        mask = np.zeros(self.row_count, dtype=bool)
        for value in values:
            normalized = _normalize_value(value, self.dtype)
            if normalized is None:
                continue
            rows = self.value_to_rows.get(normalized)
            if rows is not None:
                mask[rows] = True
        return mask

    def _range(self, op: str, value: Any) -> np.ndarray:
        mask = np.zeros(self.row_count, dtype=bool)
        normalized = _normalize_value(value, self.dtype)
        if normalized is None or self.sorted_values.size == 0:
            return mask
        if op == "<":
            end = np.searchsorted(self.sorted_values, normalized, side="left")
            rows = self.sorted_rows[:end]
        elif op == "<=":
            end = np.searchsorted(self.sorted_values, normalized, side="right")
            rows = self.sorted_rows[:end]
        elif op == ">":
            start = np.searchsorted(self.sorted_values, normalized, side="right")
            rows = self.sorted_rows[start:]
        else:
            start = np.searchsorted(self.sorted_values, normalized, side="left")
            rows = self.sorted_rows[start:]
        mask[rows] = True
        return mask


def _normalize_value(value: Any, dtype: DataType) -> Any:
    if value is None:
        return None
    if dtype == DataType.TIMESTAMPTZ:
        return parse_timestamptz(value)
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        if isinstance(value, bool):
            return int(value)
        return int(value)
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return float(value)
    if dtype == DataType.BOOL:
        return bool(value)
    if dtype == DataType.VARCHAR:
        return str(value)
    return value


def _numpy_dtype(dtype: DataType):
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.TIMESTAMPTZ):
        return np.int64
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return np.float64
    if dtype == DataType.BOOL:
        return np.bool_
    return object


def _arrow_type(dtype: DataType):
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64, DataType.TIMESTAMPTZ):
        return pa.int64()
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return pa.float64()
    if dtype == DataType.BOOL:
        return pa.bool_()
    if dtype == DataType.VARCHAR:
        return pa.string()
    raise NotImplementedError(f"unsupported scalar index field type {dtype.name}")
