"""Tests for IVF_SQ8 and HNSW_SQ quantized indexes.

Covers: build/search/save/load roundtrip + recall differential vs
BruteForce baseline. SQ8 quantization typically preserves >95% recall
on standard datasets, so we use a 0.85 threshold for robustness.
"""

import os
import tempfile
import numpy as np
import pytest

from milvus_lite.index.brute_force import BruteForceIndex
from milvus_lite.index.factory import (
    build_index_from_spec, load_index_from_spec, is_faiss_available,
)
from milvus_lite.index.spec import IndexSpec

pytestmark = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss-cpu is not installed"
)

if is_faiss_available():
    from milvus_lite.index.faiss_ivf_sq8 import FaissIvfSq8Index
    from milvus_lite.index.faiss_hnsw_sq import FaissHnswSqIndex


def _gen_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# IVF_SQ8
# ---------------------------------------------------------------------------

class TestIvfSq8:
    def test_build_and_search(self):
        vecs = _gen_vectors(200, 16, seed=1)
        idx = FaissIvfSq8Index.build(vecs, "COSINE", {"nlist": 16})
        assert idx.num_vectors == 200
        assert idx.dim == 16

        queries = _gen_vectors(3, 16, seed=2)
        ids, dists = idx.search(queries, top_k=5, params={"nprobe": 16})
        assert ids.shape == (3, 5)
        assert dists.shape == (3, 5)
        assert (ids >= 0).all()

    def test_empty_index(self):
        vecs = np.zeros((0, 8), dtype=np.float32)
        idx = FaissIvfSq8Index.build(vecs, "L2")
        queries = _gen_vectors(2, 8, seed=3)
        ids, _ = idx.search(queries, top_k=5)
        assert (ids == -1).all()

    def test_save_and_load(self):
        vecs = _gen_vectors(100, 8, seed=4)
        idx = FaissIvfSq8Index.build(vecs, "L2", {"nlist": 8})
        with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            loaded = FaissIvfSq8Index.load(path, "L2", 8)
            assert loaded.num_vectors == 100
            # Same search results after load
            q = _gen_vectors(3, 8, seed=5)
            ids_a, _ = idx.search(q, top_k=5, params={"nprobe": 8})
            ids_b, _ = loaded.search(q, top_k=5, params={"nprobe": 8})
            np.testing.assert_array_equal(ids_a, ids_b)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
    def test_recall_vs_brute_force(self, metric):
        vecs = _gen_vectors(500, 32, seed=11)
        queries = _gen_vectors(20, 32, seed=12)

        brute = BruteForceIndex.build(vecs, metric)
        sq8 = FaissIvfSq8Index.build(vecs, metric, {"nlist": 16})

        brute_ids, _ = brute.search(queries, top_k=10)
        sq8_ids, _ = sq8.search(queries, top_k=10, params={"nprobe": 16})

        total_recall = 0.0
        for q in range(20):
            brute_set = {int(i) for i in brute_ids[q] if i >= 0}
            sq8_set = {int(i) for i in sq8_ids[q] if i >= 0}
            total_recall += len(brute_set & sq8_set) / max(len(brute_set), 1)

        avg_recall = total_recall / 20
        assert avg_recall >= 0.85, (
            f"IVF_SQ8 recall@10 = {avg_recall:.3f} < 0.85 for metric={metric}"
        )

    def test_valid_mask(self):
        vecs = _gen_vectors(100, 8, seed=6)
        idx = FaissIvfSq8Index.build(vecs, "L2", {"nlist": 8})
        queries = _gen_vectors(2, 8, seed=7)
        # Mask: only even indices valid
        mask = np.zeros(100, dtype=bool)
        mask[::2] = True
        ids, _ = idx.search(queries, top_k=5,
                            valid_mask=mask, params={"nprobe": 8})
        for row in ids:
            for i in row:
                if i >= 0:
                    assert i % 2 == 0

    def test_via_factory(self):
        vecs = _gen_vectors(100, 8, seed=8)
        spec = IndexSpec(
            field_name="vec", index_type="IVF_SQ8",
            metric_type="COSINE", build_params={"nlist": 8},
        )
        idx = build_index_from_spec(spec, vecs)
        assert idx.index_type == "IVF_SQ8"

        with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            loaded = load_index_from_spec(spec, path, dim=8)
            assert loaded.index_type == "IVF_SQ8"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# HNSW_SQ
# ---------------------------------------------------------------------------

class TestHnswSq:
    def test_build_and_search(self):
        vecs = _gen_vectors(200, 16, seed=1)
        idx = FaissHnswSqIndex.build(vecs, "COSINE", {"M": 16, "efConstruction": 100})
        assert idx.num_vectors == 200
        assert idx.dim == 16

        queries = _gen_vectors(3, 16, seed=2)
        ids, dists = idx.search(queries, top_k=5, params={"ef": 64})
        assert ids.shape == (3, 5)
        assert (ids >= 0).all()

    def test_empty_index(self):
        vecs = np.zeros((0, 8), dtype=np.float32)
        idx = FaissHnswSqIndex.build(vecs, "L2")
        queries = _gen_vectors(2, 8, seed=3)
        ids, _ = idx.search(queries, top_k=5)
        assert (ids == -1).all()

    def test_save_and_load(self):
        vecs = _gen_vectors(100, 8, seed=4)
        idx = FaissHnswSqIndex.build(vecs, "L2", {"M": 16})
        with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            loaded = FaissHnswSqIndex.load(path, "L2", 8)
            assert loaded.num_vectors == 100
            q = _gen_vectors(3, 8, seed=5)
            ids_a, _ = idx.search(q, top_k=5, params={"ef": 64})
            ids_b, _ = loaded.search(q, top_k=5, params={"ef": 64})
            np.testing.assert_array_equal(ids_a, ids_b)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("metric", ["COSINE", "L2", "IP"])
    def test_recall_vs_brute_force(self, metric):
        vecs = _gen_vectors(500, 32, seed=13)
        queries = _gen_vectors(20, 32, seed=14)

        brute = BruteForceIndex.build(vecs, metric)
        hnsw_sq = FaissHnswSqIndex.build(
            vecs, metric, {"M": 32, "efConstruction": 200},
        )

        brute_ids, _ = brute.search(queries, top_k=10)
        sq_ids, _ = hnsw_sq.search(queries, top_k=10, params={"ef": 128})

        total_recall = 0.0
        for q in range(20):
            brute_set = {int(i) for i in brute_ids[q] if i >= 0}
            sq_set = {int(i) for i in sq_ids[q] if i >= 0}
            total_recall += len(brute_set & sq_set) / max(len(brute_set), 1)

        avg_recall = total_recall / 20
        assert avg_recall >= 0.85, (
            f"HNSW_SQ recall@10 = {avg_recall:.3f} < 0.85 for metric={metric}"
        )

    def test_valid_mask(self):
        vecs = _gen_vectors(100, 8, seed=6)
        idx = FaissHnswSqIndex.build(vecs, "L2", {"M": 16})
        queries = _gen_vectors(2, 8, seed=7)
        mask = np.zeros(100, dtype=bool)
        mask[::2] = True
        ids, _ = idx.search(queries, top_k=5, valid_mask=mask)
        for row in ids:
            for i in row:
                if i >= 0:
                    assert i % 2 == 0

    def test_via_factory(self):
        vecs = _gen_vectors(100, 8, seed=8)
        spec = IndexSpec(
            field_name="vec", index_type="HNSW_SQ",
            metric_type="COSINE", build_params={"M": 16},
        )
        idx = build_index_from_spec(spec, vecs)
        assert idx.index_type == "HNSW_SQ"

        with tempfile.NamedTemporaryFile(suffix=".idx", delete=False) as f:
            path = f.name
        try:
            idx.save(path)
            loaded = load_index_from_spec(spec, path, dim=8)
            assert loaded.index_type == "HNSW_SQ"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# End-to-end: Collection.create_index with SQ8
# ---------------------------------------------------------------------------

class TestE2E:
    def test_collection_create_ivf_sq8_index(self, tmp_path):
        from milvus_lite.engine.collection import Collection
        from milvus_lite.schema.types import (
            CollectionSchema, DataType, FieldSchema,
        )
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ])
        col = Collection("test", str(tmp_path / "d"), schema)
        col.insert([
            {"id": i, "vec": _gen_vectors(1, 8, seed=i)[0].tolist()}
            for i in range(50)
        ])
        col.flush()
        col.create_index("vec", {
            "index_type": "IVF_SQ8",
            "metric_type": "COSINE",
            "params": {"nlist": 8},
        })
        col.load()
        q = _gen_vectors(1, 8, seed=100)[0].tolist()
        res = col.search([q], top_k=5, metric_type="COSINE")
        assert len(res[0]) == 5
        col.close()

    def test_collection_create_hnsw_sq_index(self, tmp_path):
        from milvus_lite.engine.collection import Collection
        from milvus_lite.schema.types import (
            CollectionSchema, DataType, FieldSchema,
        )
        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=8),
        ])
        col = Collection("test", str(tmp_path / "d"), schema)
        col.insert([
            {"id": i, "vec": _gen_vectors(1, 8, seed=i)[0].tolist()}
            for i in range(50)
        ])
        col.flush()
        col.create_index("vec", {
            "index_type": "HNSW_SQ",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 100},
        })
        col.load()
        q = _gen_vectors(1, 8, seed=100)[0].tolist()
        res = col.search([q], top_k=5, metric_type="COSINE")
        assert len(res[0]) == 5
        col.close()
