"""Experiment 2: Keyword Search (tsvector/BM25) vs Semantic Search (pgvector) Evaluation."""
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import asyncpg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import init_db, close_db, get_pool
from app.search import search_documents

EVAL_QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "results_keyword_vs_semantic.json"

# Semantic variant queries to demonstrate semantic vs keyword capabilities
SEMANTIC_CHALLENGE_QUERIES = [
    {
        "query": "CUDA deep learning GPU library loading failure in python",
        "expected_parent_title": "ImportError: libcudnn when running a TensorFlow program",
        "category": "Synonyms & Paraphrasing (CUDA/GPU vs libcudnn/TensorFlow)",
    },
    {
        "query": "transform collection items into different model in reactive streams",
        "expected_parent_title": "RxJava: How to convert List of objects to List of another objects",
        "category": "Conceptual query (reactive streams vs RxJava, transform vs convert)",
    },
    {
        "query": "verify container cluster manager status from command line",
        "expected_parent_title": "How do I check that a docker host is in swarm mode?",
        "category": "Vocabulary mismatch (cluster manager vs docker swarm mode)",
    },
    {
        "query": "missing android support library class during app startup after jetpack update",
        "expected_parent_title": "ClassNotFoundException: Didn't find class \"android.support.v4.content.FileProvider\" after androidx migration",
        "category": "Descriptive problem vs raw exception (app startup / jetpack vs androidx migration)",
    },
    {
        "query": "extract DOM elements tree using bs4 python parser",
        "expected_parent_title": "Get all HTML tags with Beautiful Soup",
        "category": "Technical synonyms (DOM elements / bs4 vs HTML tags / Beautiful Soup)",
    },
]


async def setup_tsvector_and_gin_index(pool: asyncpg.Pool) -> None:
    """Add tsvector generated column and GIN index on documents table."""
    print("Setting up tsvector column and GIN index on table 'documents'...")
    async with pool.acquire() as conn:
        col_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'documents' AND column_name = 'content_tsv'
            );
            """
        )
        if not col_exists:
            await conn.execute(
                """
                ALTER TABLE documents
                ADD COLUMN content_tsv tsvector
                GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(parent_title, '') || ' ' || coalesce(content, ''))
                ) STORED;
                """
            )
            print("Added content_tsv generated column.")

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_content_tsv
            ON documents USING gin(content_tsv);
            """
        )
        print("Ensured GIN index 'idx_documents_content_tsv' exists.")


async def keyword_search(
    pool: asyncpg.Pool,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Full-text keyword search using PostgreSQL tsvector, plainto_tsquery, and ts_rank.
    """
    sql = """
        SELECT
            id,
            parent_title,
            content,
            ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
        FROM documents
        WHERE content_tsv @@ plainto_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT $2;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, query, top_k)

    return [
        {
            "id": row["id"],
            "parent_title": row["parent_title"],
            "content": row["content"],
            "score": float(row["score"]),
        }
        for row in rows
    ]


async def run_experiment():
    with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    print(f"Loaded {len(eval_queries)} standard eval queries.")

    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Database pool not available.")

    await setup_tsvector_and_gin_index(pool)

    # 1. Standard 30 eval queries
    semantic_hits = 0
    keyword_hits = 0
    total = len(eval_queries)
    comparison_records: List[Dict[str, Any]] = []

    print("\nRunning comparison on 30 evaluation queries (verbatim title)...")
    for item in eval_queries:
        query = item["query"]
        expected_title = item["expected_parent_title"].strip()

        sem_res = await search_documents(pool, query, top_k=5)
        sem_retrieved = [r["parent_title"].strip() for r in sem_res]
        sem_hit = expected_title in sem_retrieved
        if sem_hit:
            semantic_hits += 1

        kw_res = await keyword_search(pool, query, top_k=5)
        kw_retrieved = [r["parent_title"].strip() for r in kw_res]
        kw_hit = expected_title in kw_retrieved
        if kw_hit:
            keyword_hits += 1

        comparison_records.append({
            "query": query,
            "expected_parent_title": expected_title,
            "semantic_hit": sem_hit,
            "semantic_top1": sem_res[0]["parent_title"] if sem_res else None,
            "semantic_top1_score": sem_res[0]["similarity"] if sem_res else None,
            "keyword_hit": kw_hit,
            "keyword_top1": kw_res[0]["parent_title"] if kw_res else None,
            "keyword_top1_score": kw_res[0]["score"] if kw_res else None,
        })

    semantic_recall = semantic_hits / total if total > 0 else 0.0
    keyword_recall = keyword_hits / total if total > 0 else 0.0

    # 2. Challenge queries (natural language / semantic synonyms)
    print("\nEvaluating Semantic Challenge queries (Semantic HIT vs Keyword MISS)...")
    semantic_wins_examples: List[Dict[str, Any]] = []

    for ex in SEMANTIC_CHALLENGE_QUERIES:
        q = ex["query"]
        expected = ex["expected_parent_title"].strip()

        sem_res = await search_documents(pool, q, top_k=5)
        sem_retrieved = [r["parent_title"].strip() for r in sem_res]
        sem_hit = expected in sem_retrieved

        kw_res = await keyword_search(pool, q, top_k=5)
        kw_retrieved = [r["parent_title"].strip() for r in kw_res]
        kw_hit = expected in kw_retrieved

        semantic_wins_examples.append({
            "query": q,
            "category": ex["category"],
            "expected_parent_title": expected,
            "semantic_hit": sem_hit,
            "semantic_top3": sem_retrieved[:3],
            "keyword_hit": kw_hit,
            "keyword_top3": kw_retrieved[:3],
        })

    await close_db()

    print("\n" + "=" * 65)
    print("### Experiment 2 Results — Keyword vs Semantic Search")
    print("=" * 65)
    print(f"| Search Method      | Test Set                  | Hits / Total | Recall@5 |")
    print(f"|:------------------:|:-------------------------:|:------------:|:--------:|")
    print(f"| Keyword (tsvector) | Verbatim Queries (30)     | {keyword_hits:2}/{total:<2}      | {keyword_recall:.4f}   |")
    print(f"| Semantic (pgvector)| Verbatim Queries (30)     | {semantic_hits:2}/{total:<2}      | {semantic_recall:.4f}   |")
    kw_chal_hits = sum(1 for e in semantic_wins_examples if e["keyword_hit"])
    sem_chal_hits = sum(1 for e in semantic_wins_examples if e["semantic_hit"])
    print(f"| Keyword (tsvector) | Semantic Challenge (5)    | {kw_chal_hits:2}/{len(semantic_wins_examples):<2}      | {kw_chal_hits/len(semantic_wins_examples):.4f}   |")
    print(f"| Semantic (pgvector)| Semantic Challenge (5)    | {sem_chal_hits:2}/{len(semantic_wins_examples):<2}      | {sem_chal_hits/len(semantic_wins_examples):.4f}   |")
    print("=" * 65)

    print("\n### 3-5 Specific Examples where Semantic Search won (Semantic HIT, Keyword MISS):")
    for i, ex in enumerate(semantic_wins_examples, 1):
        print(f"\nExample {i} [{ex['category']}]:")
        print(f"  • Query: \"{ex['query']}\"")
        print(f"  • Expected Title: \"{ex['expected_parent_title']}\"")
        print(f"  • Semantic Result: {'HIT' if ex['semantic_hit'] else 'MISS'} -> {ex['semantic_top3']}")
        print(f"  • Keyword Result : {'HIT' if ex['keyword_hit'] else 'MISS'} -> {ex['keyword_top3']}")

    results_data = {
        "metrics_verbatim": {
            "total_queries": total,
            "keyword_hits": keyword_hits,
            "keyword_recall_at_5": round(keyword_recall, 4),
            "semantic_hits": semantic_hits,
            "semantic_recall_at_5": round(semantic_recall, 4),
        },
        "metrics_semantic_challenge": {
            "total_queries": len(semantic_wins_examples),
            "keyword_hits": kw_chal_hits,
            "keyword_recall_at_5": round(kw_chal_hits / len(semantic_wins_examples), 4),
            "semantic_hits": sem_chal_hits,
            "semantic_recall_at_5": round(sem_chal_hits / len(semantic_wins_examples), 4),
        },
        "semantic_wins_examples": semantic_wins_examples,
        "detailed_comparisons": comparison_records,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults successfully saved to '{RESULTS_PATH}'.")


if __name__ == "__main__":
    asyncio.run(run_experiment())
