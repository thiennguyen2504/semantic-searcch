"""Migration script to add so_question_id column and backfill from data/documents_dev.json."""
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import init_db, close_db, get_pool

DATA_FILE = PROJECT_ROOT / "data" / "documents_dev.json"


async def migrate_and_backfill():
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found!")
        sys.exit(1)

    print("Initializing database connection...")
    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Database pool initialization failed.")

    print("[1/3] Adding column 'so_question_id' if not exists...")
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS so_question_id BIGINT;")
    print("      Column ensured.")

    print(f"[2/3] Reading mapping from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Build unique mapping title -> so_question_id
    title_to_id = {}
    for d in docs:
        t = d.get("title", "").strip()
        so_id = d.get("so_question_id")
        if t and so_id is not None:
            title_to_id[t] = int(so_id)

    print(f"      Found {len(title_to_id)} unique title-to-id mappings.")

    print("[3/3] Backfilling so_question_id without modifying embedding vectors...")
    update_records = [(so_id, title) for title, so_id in title_to_id.items()]

    async with pool.acquire() as conn:
        # Use executemany for fast batch update
        await conn.executemany(
            "UPDATE documents SET so_question_id = $1 WHERE parent_title = $2;",
            update_records,
        )

        total_rows = await conn.fetchval("SELECT COUNT(*) FROM documents;")
        filled_rows = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE so_question_id IS NOT NULL;")
        sample_rows = await conn.fetch(
            "SELECT id, parent_title, so_question_id FROM documents WHERE so_question_id IS NOT NULL LIMIT 3;"
        )

    await close_db()

    print("\nMigration and backfill completed successfully!")
    print(f"- Total rows in 'documents': {total_rows}")
    print(f"- Rows with so_question_id populated: {filled_rows} ({(filled_rows/total_rows)*100:.1f}%)")
    print("\nSample updated rows:")
    for r in sample_rows:
        print(f"  id={r['id']}, so_question_id={r['so_question_id']}, title='{r['parent_title'][:50]}...'")


def main():
    asyncio.run(migrate_and_backfill())


if __name__ == "__main__":
    main()
