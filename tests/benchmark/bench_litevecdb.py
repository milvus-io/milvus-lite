#!/usr/bin/env python3
"""
MilvusLite performance benchmark — using the VectorDBBench Cohere 100K dataset.

Dataset: Cohere 100K (768 dimensions, COSINE, with ground truth)

Test items:
  1. Insert throughput (100K records, batch insert)
  2. Index build time (HNSW)
  3. Search QPS (single thread, nq=1)
  4. Search batch QPS (nq=10)
  5. Search Recall@10 (compared with ground truth)
  6. Search latency distribution (P50/P95/P99)
  7. Search performance with filter
"""

from __future__ import annotations

import shutil
import statistics
import tempfile
import time

import numpy as np
from pymilvus import MilvusClient, DataType

from milvus_lite.adapter.grpc.server import start_server_in_thread


# ── Data Loading ──────────────────────────────────────────────────────

def load_cohere_100k():
    """Load the Cohere 100K dataset from VectorDBBench cache"""
    from vectordb_bench.backend.dataset import Dataset
    from vectordb_bench.backend.data_source import DatasetSource

    dm = Dataset.COHERE.manager(100_000)
    if not dm.data_dir.exists():
        print("Downloading Cohere 100K dataset...")
        dm.prepare(source=DatasetSource.AliyunOSS)

    import pyarrow.parquet as pq

    # Training data (100K vectors)
    train_path = dm.data_dir / dm.data.train_files[0]
    train_table = pq.read_table(train_path)
    train_ids = train_table.column("id").to_pylist()
    train_vecs = np.array(train_table.column("emb").to_pylist(), dtype=np.float32)

    # Test data (query vectors)
    test_path = dm.data_dir / dm.data.test_file
    test_table = pq.read_table(test_path)
    test_vecs = np.array(test_table.column("emb").to_pylist(), dtype=np.float32)

    # Ground truth
    gt_path = dm.data_dir / "neighbors.parquet"
    gt_table = pq.read_table(gt_path)
    gt_neighbors = gt_table.column("neighbors_id").to_pylist()

    print(f"Loaded: train={len(train_ids)}, test={len(test_vecs)}, "
          f"dim={train_vecs.shape[1]}")
    return train_ids, train_vecs, test_vecs, gt_neighbors


# ── Utility Functions ─────────────────────────────────────────────────

def percentile(data, p):
    """Compute percentile"""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def compute_recall(predicted: list[int], ground_truth: list[int], k: int) -> float:
    """Compute Recall@K"""
    gt_set = set(ground_truth[:k])
    pred_set = set(predicted[:k])
    return len(gt_set & pred_set) / k


# ── Main Benchmark ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MilvusLite Performance Benchmark — Cohere 100K (768d, COSINE)")
    print("=" * 70)

    # 1. Load data
    print("\n[1/7] Loading dataset...")
    train_ids, train_vecs, test_vecs, gt_neighbors = load_cohere_100k()
    n_train = len(train_ids)
    dim = train_vecs.shape[1]
    n_test = len(test_vecs)

    # 2. Start server
    print("\n[2/7] Starting MilvusLite server...")
    data_dir = tempfile.mkdtemp(prefix="bench_milvus_lite_")
    server, db, port = start_server_in_thread(data_dir)
    client = MilvusClient(uri=f"http://127.0.0.1:{port}")

    try:
        # 3. Create collection (without index, insert first)
        print("\n[3/7] Creating collection...")
        schema = client.create_schema()
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)

        client.create_collection("bench", schema=schema)

        # 4. Insert data
        print(f"\n[4/7] Inserting {n_train} records (batch_size=1000)...")
        batch_size = 1000
        t_insert_start = time.perf_counter()

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch = [
                {"pk": int(train_ids[i]), "vec": train_vecs[i].tolist()}
                for i in range(start, end)
            ]
            client.insert("bench", batch)

            if (start // batch_size) % 20 == 0:
                elapsed = time.perf_counter() - t_insert_start
                print(f"  ... {end}/{n_train} ({elapsed:.1f}s)")

        t_insert_end = time.perf_counter()
        insert_time = t_insert_end - t_insert_start
        insert_qps = n_train / insert_time
        print(f"  Insert done: {insert_time:.2f}s, {insert_qps:.0f} records/s")

        # 5. Create index
        print("\n[5/7] Creating HNSW index...")
        t_index_start = time.perf_counter()

        idx = client.prepare_index_params()
        idx.add_index(field_name="vec", index_type="HNSW", metric_type="COSINE",
                      params={"M": 16, "efConstruction": 200})
        client.create_index("bench", idx)
        client.load_collection("bench")

        t_index_end = time.perf_counter()
        index_time = t_index_end - t_index_start
        print(f"  Index + Load done: {index_time:.2f}s")

        # 6. Search performance (single query)
        print(f"\n[6/7] Search benchmark (nq=1, top_k=10, {n_test} queries)...")
        top_k = 10
        latencies = []
        all_results = []

        # Warmup
        for i in range(min(10, n_test)):
            client.search("bench", data=[test_vecs[i].tolist()], limit=top_k,
                          search_params={"ef": 128}, output_fields=["pk"])

        # Benchmark
        t_search_start = time.perf_counter()
        for i in range(n_test):
            t0 = time.perf_counter()
            results = client.search("bench", data=[test_vecs[i].tolist()],
                                    limit=top_k,
                                    search_params={"ef": 128},
                                    output_fields=["pk"])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
            all_results.append([h["entity"]["pk"] for h in results[0]])

        t_search_end = time.perf_counter()
        total_search_time = t_search_end - t_search_start
        search_qps = n_test / total_search_time

        print(f"  Total: {total_search_time:.2f}s")
        print(f"  QPS (nq=1): {search_qps:.1f}")
        print(f"  Latency P50: {percentile(latencies, 50):.2f} ms")
        print(f"  Latency P95: {percentile(latencies, 95):.2f} ms")
        print(f"  Latency P99: {percentile(latencies, 99):.2f} ms")
        print(f"  Latency avg: {statistics.mean(latencies):.2f} ms")

        # 7. Recall
        print(f"\n[7/7] Computing Recall@{top_k}...")
        recalls = []
        for i in range(n_test):
            if i < len(gt_neighbors) and gt_neighbors[i]:
                r = compute_recall(all_results[i], gt_neighbors[i], top_k)
                recalls.append(r)

        avg_recall = statistics.mean(recalls) if recalls else 0
        print(f"  Recall@{top_k}: {avg_recall:.4f} ({avg_recall*100:.2f}%)")

        # ── Bonus: batch query ──
        # Note:
        #   batch_rps = RPC calls per second (strict QPS definition, typically lower than nq=1 QPS when nq=10)
        #   batch_vector_throughput = total vector queries processed per second (throughput perspective)
        print(f"\n[Bonus] Batch search (nq=10)...")
        batch_latencies = []
        n_batches = n_test // 10
        for i in range(n_batches):
            batch_q = test_vecs[i*10:(i+1)*10].tolist()
            t0 = time.perf_counter()
            client.search("bench", data=batch_q, limit=top_k,
                          search_params={"ef": 128}, output_fields=["pk"])
            t1 = time.perf_counter()
            batch_latencies.append((t1 - t0) * 1000)

        total_batch = sum(batch_latencies) / 1000
        batch_rps = n_batches / total_batch
        batch_vec_throughput = (n_batches * 10) / total_batch
        print(f"  RPS (nq=10, per-RPC):       {batch_rps:.1f}")
        print(f"  Vector throughput (nq=10):  {batch_vec_throughput:.1f} vec/s")
        print(f"  Batch latency avg:          {statistics.mean(batch_latencies):.2f} ms")

        # ── Summary ──
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Dataset:         Cohere 100K ({dim}d, COSINE)")
        print(f"  Insert:          {n_train} records in {insert_time:.2f}s "
              f"({insert_qps:.0f} rec/s)")
        print(f"  Index (HNSW):    {index_time:.2f}s "
              f"(M=16, efConstruction=200)")
        print(f"  QPS (nq=1):      {search_qps:.1f} (top10, ef=128)")
        print(f"  RPS (nq=10):     {batch_rps:.1f} (batched RPC throughput)")
        print(f"  Vec throughput:  {batch_vec_throughput:.1f} vec/s (nq=10 batched)")
        print(f"  Latency P50:     {percentile(latencies, 50):.2f} ms")
        print(f"  Latency P95:     {percentile(latencies, 95):.2f} ms")
        print(f"  Latency P99:     {percentile(latencies, 99):.2f} ms")
        print(f"  Recall@10:       {avg_recall*100:.2f}%")
        print("=" * 70)

    finally:
        client.close()
        server.stop(grace=2)
        db.close()
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
