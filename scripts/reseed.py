"""Reseed database with new multilingual embedding model using data/documents_dev.json."""
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
from app.embedding import MODEL_NAME

DATA_FILE = PROJECT_ROOT / "data" / "documents_dev.json"


async def run_reseed():
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Please run scripts/prepare_data.py first.")
        sys.exit(1)

    print(f"============================================================")
    print(f"Reseeding database with embedding model: {MODEL_NAME}")
    print(f"============================================================")
    print(f"Loading documents from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Total documents to re-seed: {len(documents)}")

    print("Initializing database connection pool...")
    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Failed to initialize database pool.")

    print("Truncating 'documents' table to clear old vector embeddings...")
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
    print("Table truncated.")

    total_chunks_inserted = 0
    print("Starting insertion with new multilingual embeddings...")
    for doc in tqdm(documents, desc="Reseeding documents", unit="doc"):
        title = doc.get("title", "")
        content = doc.get("content", "")
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

    print("\nReseeding finished successfully!")
    print(f"- Total chunks inserted: {total_chunks_inserted}")
    print(f"- Total rows in 'documents' table: {db_total_count}")


def main():
    asyncio.run(run_reseed())


if __name__ == "__main__":
    main()
