"""Performance benchmark — insert, search, query at 10K-50K scale.

Marked as ``slow`` so it doesn't run in normal CI. Run explicitly:

    pytest tests/benchmark/test_performance.py -v -s

Reports throughput (rows/s) and latency (ms) to stdout.
"""

import tempfile
import time
from typing import List

import numpy as np
import pytest

from milvus_lite.schema.types import CollectionSchema, DataType, FieldSchema
from milvus_lite.engine.collection import Collection
from milvus_lite.index.factory import is_faiss_available

pytestmark = pytest.mark.slow

DIM = 128
SEED = 42


def _gen_vectors(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make_records(n: int, dim: int = DIM) -> List[dict]:
    vectors = _gen_vectors(n, dim)
    return [
        {
            "id": i,
            "vec": vectors[i].tolist(),
            "category": f"cat_{i % 10}",
            "score": float(i % 100),
        }
        for i in range(n)
    ]


def _make_schema(dim: int = DIM) -> CollectionSchema:
    return CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="score", dtype=DataType.FLOAT),
    ])


def _report(label: str, elapsed: float, count: int = 1):
    if count > 1:
        throughput = count / elapsed
        print(f"  {label}: {elapsed*1000:.1f}ms ({throughput:.0f} rows/s)")
    else:
        print(f"  {label}: {elapsed*1000:.1f}ms")


# ---------------------------------------------------------------------------
# Insert benchmark
# ---------------------------------------------------------------------------

class TestInsertPerformance:

    @pytest.mark.parametrize("n", [1000, 10000, 50000])
    def test_insert_throughput(self, n):
        """Measure bulk insert throughput."""
        records = _make_records(n)
        print(f"\n=== Insert {n} rows (dim={DIM}) ===")

        with tempfile.TemporaryDirectory() as d:
            col = Collection("bench", d, _make_schema())

            t0 = time.perf_counter()
            col.insert(records)
            elapsed = time.perf_counter() - t0

            _report("insert", elapsed, n)
            col.close()


# ---------------------------------------------------------------------------
# Search benchmark
# ---------------------------------------------------------------------------

class TestSearchPerformance:

    def _setup_collection(self, tmpdir, n, index_type, metric="COSINE"):
        records = _make_records(n)
        col = Collection("bench", tmpdir, _make_schema())
        col.insert(records)
        col.create_index("vec", {
            "index_type": index_type,
            "metric_type": metric,
            "params": {"M": 16, "efConstruction": 200} if index_type == "HNSW" else
                      {"nlist": 64} if index_type == "IVF_FLAT" else {},
        })
        col.load()
        return col

    @pytest.mark.parametrize("index_type", ["FLAT", "HNSW", "IVF_FLAT"])
    def test_single_search(self, index_type):
        """Single-query search latency across index types."""
        if index_type in ("HNSW", "IVF_FLAT") and not is_faiss_available():
            pytest.skip("faiss-cpu not installed")

        n = 50000
        nq = 1
        top_k = 10
        query = _gen_vectors(nq, DIM)

        print(f"\n=== Single search: {index_type}, {n} rows, top_k={top_k} ===")

        with tempfile.TemporaryDirectory() as d:
            col = self._setup_collection(d, n, index_type)

            # Warm up
            col.search(query.tolist(), top_k=top_k)

            # Benchmark: 10 iterations
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                results = col.search(query.tolist(), top_k=top_k)
                times.append(time.perf_counter() - t0)

            assert len(results[0]) == top_k

            avg = sum(times) / len(times)
            p50 = sorted(times)[len(times) // 2]
            p99 = sorted(times)[int(len(times) * 0.99)]
            print(f"  avg: {avg*1000:.1f}ms, p50: {p50*1000:.1f}ms, p99: {p99*1000:.1f}ms")
            col.close()

    @pytest.mark.parametrize("index_type", ["FLAT", "HNSW"])
    def test_batch_search(self, index_type):
        """Batch search (nq=100) latency."""
        if index_type == "HNSW" and not is_faiss_available():
            pytest.skip("faiss-cpu not installed")

        n = 50000
        nq = 100
        top_k = 10
        queries = _gen_vectors(nq, DIM)

        print(f"\n=== Batch search: {index_type}, {n} rows, nq={nq}, top_k={top_k} ===")

        with tempfile.TemporaryDirectory() as d:
            col = self._setup_collection(d, n, index_type)

            t0 = time.perf_counter()
            results = col.search(queries.tolist(), top_k=top_k)
            elapsed = time.perf_counter() - t0

            assert len(results) == nq
            _report("batch search", elapsed)
            _report("per query", elapsed / nq)
            col.close()

    def test_search_with_filter(self):
        """Search with scalar filter vs without."""
        n = 50000
        top_k = 10
        query = _gen_vectors(1, DIM)

        print(f"\n=== Search with filter: FLAT, {n} rows ===")

        with tempfile.TemporaryDirectory() as d:
            col = self._setup_collection(d, n, "FLAT")

            # Without filter
            times_no_filter = []
            for _ in range(10):
                t0 = time.perf_counter()
                col.search(query.tolist(), top_k=top_k)
                times_no_filter.append(time.perf_counter() - t0)

            # With filter (selects ~10% of rows)
            times_with_filter = []
            for _ in range(10):
                t0 = time.perf_counter()
                col.search(query.tolist(), top_k=top_k,
                           expr='category == "cat_0"')
                times_with_filter.append(time.perf_counter() - t0)

            avg_no = sum(times_no_filter) / len(times_no_filter)
            avg_with = sum(times_with_filter) / len(times_with_filter)
            print(f"  without filter: {avg_no*1000:.1f}ms")
            print(f"  with filter:    {avg_with*1000:.1f}ms")
            col.close()


# ---------------------------------------------------------------------------
# Query benchmark
# ---------------------------------------------------------------------------

class TestQueryPerformance:

    def test_query_throughput(self):
        """Scalar query latency."""
        n = 50000
        print(f"\n=== Query: {n} rows ===")

        with tempfile.TemporaryDirectory() as d:
            col = Collection("bench", d, _make_schema())
            col.insert(_make_records(n))
            col.load()

            # Simple filter (~10% selectivity)
            t0 = time.perf_counter()
            rows = col.query('category == "cat_0"', limit=1000)
            elapsed = time.perf_counter() - t0
            _report(f"category filter ({len(rows)} rows)", elapsed)

            # Range filter (~50% selectivity)
            t0 = time.perf_counter()
            rows = col.query("score >= 50", limit=10000)
            elapsed = time.perf_counter() - t0
            _report(f"range filter ({len(rows)} rows)", elapsed)

            # Complex filter
            t0 = time.perf_counter()
            rows = col.query('category == "cat_0" and score >= 50', limit=1000)
            elapsed = time.perf_counter() - t0
            _report(f"compound filter ({len(rows)} rows)", elapsed)

            col.close()


# ---------------------------------------------------------------------------
# Index build benchmark
# ---------------------------------------------------------------------------

class TestIndexBuildPerformance:

    @pytest.mark.parametrize("index_type", ["FLAT", "HNSW", "IVF_FLAT"])
    def test_index_build_time(self, index_type):
        """Measure index build + load time."""
        if index_type in ("HNSW", "IVF_FLAT") and not is_faiss_available():
            pytest.skip("faiss-cpu not installed")

        n = 50000
        print(f"\n=== Index build: {index_type}, {n} rows (dim={DIM}) ===")

        with tempfile.TemporaryDirectory() as d:
            col = Collection("bench", d, _make_schema())
            col.insert(_make_records(n))

            col.create_index("vec", {
                "index_type": index_type,
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200} if index_type == "HNSW" else
                          {"nlist": 64} if index_type == "IVF_FLAT" else {},
            })

            t0 = time.perf_counter()
            col.load()
            elapsed = time.perf_counter() - t0

            _report("build + load", elapsed)
            col.close()
