"""Seed documents into PostgreSQL from data/documents_dev.json."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import init_db, close_db, get_pool
from app.search import insert_document

DATA_FILE = PROJECT_ROOT / "data" / "documents_dev.json"


async def run_seed(limit: int | None = None, clear: bool = False):
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Please run scripts/prepare_data.py first.")
        sys.exit(1)

    print(f"Loading documents from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    if limit is not None and limit > 0:
        documents = documents[:limit]
        print(f"Limited seeding to first {len(documents)} documents.")
    else:
        print(f"Total documents to seed: {len(documents)}")

    print("Initializing database connection pool...")
    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Failed to initialize database pool.")

    if clear:
        print("Clearing existing documents from database...")
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
        print("Documents table truncated.")

    total_chunks_inserted = 0

    print("Starting insertion...")
    for doc in tqdm(documents, desc="Seeding documents", unit="doc"):
        title = doc.get("title", "")
        content = doc.get("content", "")
        if not content.strip():
            continue
        so_question_id = doc.get("so_question_id")
        chunks_count = await insert_document(
            pool,
            title,
            content,
            so_question_id=so_question_id,
        )
        total_chunks_inserted += chunks_count

    # Check total rows in the documents table
    async with pool.acquire() as conn:
        db_total_count = await conn.fetchval("SELECT COUNT(*) FROM documents;")

    await close_db()

    print("\nSeeding finished!")
    print(f"- Chunks inserted in this run: {total_chunks_inserted}")
    print(f"- Total chunks in database ('documents' table): {db_total_count}")


def main():
    parser = argparse.ArgumentParser(description="Seed documents into database.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of documents to seed (optional, for quick testing)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before seeding",
    )
    args = parser.parse_args()

    asyncio.run(run_seed(limit=args.limit, clear=args.clear))



if __name__ == "__main__":
    main()
