"""Experiment 3: Vector Index Latency Benchmark (with vs without HNSW)."""
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import asyncpg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import init_db, close_db, get_pool
from app.chunking import chunk_text
from app.embedding import embed, embed_batch

FULL_DOCS_PATH = PROJECT_ROOT / "data" / "documents_full.json"
EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"
RESULTS_JSON_PATH = PROJECT_ROOT / "data" / "results_index_latency.json"
CHART_PNG_PATH = PROJECT_ROOT / "data" / "latency_chart.png"

TABLE_NAME = "documents_bench"


async def setup_bench_table(pool: asyncpg.Pool):
    """Create benchmark table."""
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                parent_title TEXT,
                chunk_index INT,
                content TEXT,
                embedding VECTOR(384)
            );
            TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY;
            DROP INDEX IF EXISTS idx_bench_hnsw;
            """
        )


async def insert_docs_slice(
    pool: asyncpg.Pool,
    docs_slice: List[Dict[str, str]],
    start_offset: int,
):
    """Insert a slice of documents into the benchmark table."""
    chunk_records = []
    for doc in docs_slice:
        title = doc.get("title", "")
        content = doc.get("content", "")
        if not content.strip():
            continue
        chunks = chunk_text(content, chunk_size=300, overlap=50)
        for idx, ch in enumerate(chunks):
            chunk_records.append((title, idx, ch))

    total = len(chunk_records)
    batch_size = 128
    insert_sql = f"""
        INSERT INTO {TABLE_NAME} (parent_title, chunk_index, content, embedding)
        VALUES ($1, $2, $3, $4);
    """

    for i in range(0, total, batch_size):
        batch = chunk_records[i : i + batch_size]
        batch_texts = [item[2] for item in batch]
        embeddings = embed_batch(batch_texts, batch_size=batch_size)

        db_records = [
            (item[0], item[1], item[2], np.array(emb, dtype=np.float32))
            for item, emb in zip(batch, embeddings)
        ]
        async with pool.acquire() as conn:
            await conn.executemany(insert_sql, db_records)


async def measure_avg_latency(
    pool: asyncpg.Pool,
    query_vectors: List[np.ndarray],
    repetitions: int = 3,
) -> float:
    """Measure average query latency in milliseconds over all query vectors."""
    search_sql = f"""
        SELECT id
        FROM {TABLE_NAME}
        ORDER BY embedding <=> $1
        LIMIT 5;
    """

    latencies_ms = []

    async with pool.acquire() as conn:
        # Warmup with first query
        await conn.fetch(search_sql, query_vectors[0])

        for q_vec in query_vectors:
            q_times = []
            for _ in range(repetitions):
                t0 = time.perf_counter()
                await conn.fetch(search_sql, q_vec)
                t1 = time.perf_counter()
                q_times.append((t1 - t0) * 1000.0)
            latencies_ms.append(sum(q_times) / len(q_times))

    return float(np.mean(latencies_ms))


def plot_results(benchmarks: List[Dict[str, Any]], output_path: Path):
    """Plot latency comparison chart and save to PNG."""
    doc_counts = [b["doc_count"] for b in benchmarks]
    no_index_latencies = [b["latency_no_index_ms"] for b in benchmarks]
    hnsw_latencies = [b["latency_hnsw_ms"] for b in benchmarks]

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(
        doc_counts,
        no_index_latencies,
        marker="o",
        linewidth=2.5,
        color="#e63946",
        label="Without Index (Flat Sequential Scan)",
    )
    plt.plot(
        doc_counts,
        hnsw_latencies,
        marker="s",
        linewidth=2.5,
        color="#2a9d8f",
        label="With HNSW Index (Hierarchical Navigable Small World)",
    )

    for x, y in zip(doc_counts, no_index_latencies):
        plt.annotate(
            f"{y:.2f} ms",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            color="#e63946",
        )

    for x, y in zip(doc_counts, hnsw_latencies):
        plt.annotate(
            f"{y:.2f} ms",
            (x, y),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontweight="bold",
            color="#2a9d8f",
        )

    plt.title("Semantic Search Query Latency: Flat Scan vs HNSW Index", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Number of Documents", fontsize=11, labelpad=10)
    plt.ylabel("Average Latency (ms)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Chart saved to {output_path}")


async def run_experiment():
    with open(FULL_DOCS_PATH, "r", encoding="utf-8") as f:
        full_documents = json.load(f)

    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    total_available = len(full_documents)
    print(f"Total available full documents: {total_available}")
    print(f"Loaded {len(eval_queries)} eval queries.")

    # Determine milestones
    milestones = [5000, 10000, total_available]
    # Filter milestones within available documents
    milestones = sorted(list(set([m for m in milestones if m <= total_available])))
    print(f"Benchmark milestones: {milestones}")

    # Pre-embed the 30 eval queries
    print("Pre-embedding 30 evaluation queries...")
    query_vectors = [np.array(embed(q["query"]), dtype=np.float32) for q in eval_queries]

    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Database pool not available.")

    await setup_bench_table(pool)

    benchmarks: List[Dict[str, Any]] = []
    current_docs_count = 0

    for milestone in milestones:
        needed = milestone - current_docs_count
        docs_slice = full_documents[current_docs_count:milestone]
        print(f"\n==========================================")
        print(f"Milestone: {milestone} documents (+{needed} new documents)")
        print(f"==========================================")

        print(f"Seeding documents from index {current_docs_count} to {milestone}...")
        t0_seed = time.perf_counter()
        await insert_docs_slice(pool, docs_slice, current_docs_count)
        current_docs_count = milestone
        print(f"Seeded in {time.perf_counter() - t0_seed:.2f}s.")

        async with pool.acquire() as conn:
            chunk_count = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        print(f"Current total chunks in table: {chunk_count}")

        # 1. Drop HNSW index to measure unindexed flat scan latency
        print("Measuring latency WITHOUT HNSW index (flat scan)...")
        async with pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_bench_hnsw;")
        latency_no_index = await measure_avg_latency(pool, query_vectors, repetitions=3)
        print(f"  -> Latency WITHOUT index: {latency_no_index:.2f} ms")

        # 2. Build HNSW index
        print("Building HNSW index (vector_cosine_ops)...")
        t0_idx = time.perf_counter()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE INDEX idx_bench_hnsw
                ON {TABLE_NAME}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        idx_build_time = time.perf_counter() - t0_idx
        print(f"  -> HNSW index built in {idx_build_time:.2f}s.")

        # 3. Measure latency with HNSW index
        print("Measuring latency WITH HNSW index...")
        latency_hnsw = await measure_avg_latency(pool, query_vectors, repetitions=3)
        print(f"  -> Latency WITH HNSW index: {latency_hnsw:.2f} ms")

        speedup = (latency_no_index / latency_hnsw) if latency_hnsw > 0 else 1.0
        print(f"  -> Speedup: {speedup:.1f}x faster")

        benchmarks.append({
            "doc_count": milestone,
            "chunk_count": chunk_count,
            "latency_no_index_ms": round(latency_no_index, 2),
            "latency_hnsw_ms": round(latency_hnsw, 2),
            "speedup": round(speedup, 2),
            "index_build_time_sec": round(idx_build_time, 2),
        })

    # Cleanup bench table to save space
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")

    await close_db()

    # Save results to JSON
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved latency results to {RESULTS_JSON_PATH}")

    # Plot chart
    plot_results(benchmarks, CHART_PNG_PATH)

    # Print summary markdown table
    print("\n" + "=" * 70)
    print("### Experiment 3 Results: Vector Index Latency Benchmark")
    print("=" * 70)
    print("| Documents   | Total Chunks | No Index Latency (ms) | HNSW Latency (ms) | Speedup          |")
    print("|:-----------:|:------------:|:---------------------:|:-----------------:|:----------------:|")
    for b in benchmarks:
        print(f"| {b['doc_count']:11} | {b['chunk_count']:12} | {b['latency_no_index_ms']:21.2f} | {b['latency_hnsw_ms']:17.2f} | {b['speedup']:14.1f}x |")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_experiment())
