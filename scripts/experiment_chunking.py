"""Experiment 1: Chunking Configurations Evaluation (Recall@5)."""
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import asyncpg
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import init_db, close_db, get_pool
from app.chunking import chunk_text
from app.embedding import embed, embed_batch

DEV_DOCS_PATH = PROJECT_ROOT / "data" / "documents_dev.json"
EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "results_chunking.json"

CONFIGS: List[Tuple[int, int]] = [
    (300, 50),
    (500, 50),
    (500, 100),
]


async def create_table_for_config(pool: asyncpg.Pool, table_name: str) -> None:
    """Create table if not exists, and clear existing rows."""
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                parent_title TEXT,
                chunk_index INT,
                content TEXT,
                embedding VECTOR(384)
            );
            TRUNCATE TABLE {table_name} RESTART IDENTITY;
            """
        )


async def seed_config_table(
    pool: asyncpg.Pool,
    documents: List[Dict[str, str]],
    chunk_size: int,
    overlap: int,
    table_name: str,
) -> int:
    """Chunk and embed documents with specific config, inserting into table_name."""
    print(f"\n--- Seeding table '{table_name}' with chunk_size={chunk_size}, overlap={overlap} ---")

    # Collect all chunks across all documents
    chunk_records: List[Tuple[str, int, str]] = []
    for doc in documents:
        title = doc.get("title", "")
        content = doc.get("content", "")
        if not content.strip():
            continue
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            chunk_records.append((title, idx, ch))

    total_chunks = len(chunk_records)
    print(f"Total chunks generated: {total_chunks} from {len(documents)} documents.")

    # Embed chunks in batches for efficiency
    batch_size = 128
    insert_sql = f"""
        INSERT INTO {table_name} (parent_title, chunk_index, content, embedding)
        VALUES ($1, $2, $3, $4);
    """

    for i in tqdm(range(0, total_chunks, batch_size), desc=f"Embedding & inserting {table_name}"):
        batch = chunk_records[i : i + batch_size]
        batch_texts = [item[2] for item in batch]
        batch_embeddings = embed_batch(batch_texts, batch_size=batch_size)

        db_records = [
            (item[0], item[1], item[2], np.array(emb, dtype=np.float32))
            for item, emb in zip(batch, batch_embeddings)
        ]
        async with pool.acquire() as conn:
            await conn.executemany(insert_sql, db_records)

    return total_chunks


async def evaluate_recall_at_5(
    pool: asyncpg.Pool,
    eval_queries: List[Dict[str, str]],
    table_name: str,
) -> Tuple[float, int, int]:
    """
    Compute Recall@5 on eval_queries against table_name.
    Hit if expected_parent_title is in top-5 retrieved parent_titles.
    """
    hits = 0
    total = len(eval_queries)

    search_sql = f"""
        SELECT parent_title, 1 - (embedding <=> $1) as similarity
        FROM {table_name}
        ORDER BY embedding <=> $1
        LIMIT 5;
    """

    for item in eval_queries:
        query = item["query"]
        expected_title = item["expected_parent_title"].strip()
        query_vector = np.array(embed(query), dtype=np.float32)

        async with pool.acquire() as conn:
            rows = await conn.fetch(search_sql, query_vector)

        retrieved_titles = [r["parent_title"].strip() for r in rows]
        if expected_title in retrieved_titles:
            hits += 1

    recall = hits / total if total > 0 else 0.0
    return recall, hits, total


async def run_experiment():
    with open(DEV_DOCS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    print(f"Loaded {len(documents)} documents and {len(eval_queries)} eval queries.")

    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Database pool not available.")

    results: List[Dict[str, Any]] = []

    for chunk_size, overlap in CONFIGS:
        table_name = f"documents_c{chunk_size}_{overlap}"
        await create_table_for_config(pool, table_name)

        # Optimization: If (300, 50), check if we can copy from seeded 'documents' table
        if (chunk_size, overlap) == (300, 50):
            async with pool.acquire() as conn:
                existing_count = await conn.fetchval("SELECT COUNT(*) FROM documents;")
            if existing_count > 0:
                print(f"Copying {existing_count} pre-seeded chunks from 'documents' into '{table_name}'...")
                async with pool.acquire() as conn:
                    await conn.execute(f"INSERT INTO {table_name} SELECT * FROM documents;")
                total_chunks = existing_count
            else:
                total_chunks = await seed_config_table(pool, documents, chunk_size, overlap, table_name)
        else:
            total_chunks = await seed_config_table(pool, documents, chunk_size, overlap, table_name)

        recall, hits, total = await evaluate_recall_at_5(pool, eval_queries, table_name)
        print(f"Config ({chunk_size}, {overlap}): Hits = {hits}/{total}, Recall@5 = {recall:.4f}")

        results.append({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "config": f"({chunk_size}, {overlap})",
            "table_name": table_name,
            "total_chunks": total_chunks,
            "hits": hits,
            "total_queries": total,
            "recall_at_5": round(recall, 4),
        })

    await close_db()

    # Save results to json
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    # Print markdown table
    print("\n" + "=" * 50)
    print("### Experiment 1 Results — Chunking Comparison")
    print("=" * 50)
    print("| Config (chunk_size, overlap) | Total Chunks | Hits / Total | Recall@5 |")
    print("|:----------------------------:|:------------:|:------------:|:--------:|")
    for r in results:
        print(f"| {r['config']:28} | {r['total_chunks']:12} | {r['hits']}/{r['total_queries']}        | {r['recall_at_5']:.4f}   |")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_experiment())
