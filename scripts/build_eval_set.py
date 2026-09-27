"""Build evaluation query set from dev documents."""
import json
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
DEV_DOCS_PATH = OUTPUT_DIR / "documents_dev.json"
EVAL_QUERIES_PATH = OUTPUT_DIR / "eval_queries.json"


def main():
    if not DEV_DOCS_PATH.exists():
        raise FileNotFoundError(f"Dev documents not found at: {DEV_DOCS_PATH}. Run scripts/prepare_data.py first.")

    print(f"Reading documents from '{DEV_DOCS_PATH}'...")
    with open(DEV_DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} documents.")

    sample_size = min(30, len(docs))
    # Reproducible sampling
    rng = random.Random(42)
    sampled_docs = rng.sample(docs, sample_size)

    eval_queries = []
    for doc in sampled_docs:
        title = doc["title"].strip()
        eval_queries.append({
            "query": title,
            "expected_parent_title": title,
        })

    with open(EVAL_QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_queries, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated {len(eval_queries)} evaluation queries to '{EVAL_QUERIES_PATH}'.")
    if eval_queries:
        print("\nSample query preview:")
        print(f"- Query: {eval_queries[0]['query']}")
        print(f"- Expected Title: {eval_queries[0]['expected_parent_title']}")


if __name__ == "__main__":
    main()
