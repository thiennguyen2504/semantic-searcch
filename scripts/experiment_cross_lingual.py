"""Experiment Cross-Lingual: Evaluate Recall@5 for English vs Vietnamese queries."""
import asyncio
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import init_db, close_db, get_pool
from app.search import search_documents

EN_EVAL_FILE = PROJECT_ROOT / "data" / "eval_queries.json"
VI_EVAL_FILE = PROJECT_ROOT / "data" / "eval_queries_vi.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results_cross_lingual.json"


async def evaluate_queries(pool, eval_file: Path, lang_name: str):
    if not eval_file.exists():
        raise FileNotFoundError(f"File {eval_file} does not exist.")

    with open(eval_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    hits = 0
    detailed_results = []

    for item in queries:
        query_text = item["query"]
        expected_title = item["expected_parent_title"]

        results = await search_documents(pool, query_text, top_k=5)

        is_hit = False
        hit_rank = None
        for rank, res in enumerate(results, start=1):
            if res["parent_title"].strip() == expected_title.strip():
                is_hit = True
                hit_rank = rank
                break

        if is_hit:
            hits += 1

        detailed_results.append({
            "query": query_text,
            "expected_parent_title": expected_title,
            "is_hit": is_hit,
            "hit_rank": hit_rank,
            "top_5": [
                {
                    "rank": i + 1,
                    "parent_title": r["parent_title"],
                    "similarity": round(r["similarity"], 4),
                    "content_snippet": r["content"][:200] + ("..." if len(r["content"]) > 200 else "")
                }
                for i, r in enumerate(results)
            ]
        })

    total = len(queries)
    recall_at_5 = hits / total if total > 0 else 0.0

    return {
        "language": lang_name,
        "total_queries": total,
        "hits": hits,
        "recall_at_5": round(recall_at_5, 4),
        "detailed_results": detailed_results,
    }


async def run_experiment():
    print("Initializing database connection pool...")
    await init_db()
    pool = get_pool()
    if pool is None:
        raise RuntimeError("Failed to initialize database pool.")

    print("\n--- Running English Evaluation Queries ---")
    en_results = await evaluate_queries(pool, EN_EVAL_FILE, "English")
    print(f"English Recall@5: {en_results['hits']}/{en_results['total_queries']} = {en_results['recall_at_5']:.4f}")

    print("\n--- Running Vietnamese Evaluation Queries ---")
    vi_results = await evaluate_queries(pool, VI_EVAL_FILE, "Vietnamese")
    print(f"Vietnamese Recall@5: {vi_results['hits']}/{vi_results['total_queries']} = {vi_results['recall_at_5']:.4f}")

    await close_db()

    # Separate hits and misses in Vietnamese queries
    vi_hits = [r for r in vi_results["detailed_results"] if r["is_hit"]]
    vi_misses = [r for r in vi_results["detailed_results"] if not r["is_hit"]]

    output_data = {
        "summary": {
            "english": {
                "total": en_results["total_queries"],
                "hits": en_results["hits"],
                "recall_at_5": en_results["recall_at_5"]
            },
            "vietnamese": {
                "total": vi_results["total_queries"],
                "hits": vi_results["hits"],
                "recall_at_5": vi_results["recall_at_5"]
            }
        },
        "vietnamese_hits": vi_hits,
        "vietnamese_misses": vi_misses,
        "all_english_results": en_results["detailed_results"],
        "all_vietnamese_results": vi_results["detailed_results"],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cross-lingual results to {OUTPUT_FILE}")

    # Print summary Markdown table
    print("\n" + "=" * 60)
    print("BẢNG KẾT QUẢ RECALL@5 (CROSS-LINGUAL SEARCH)")
    print("=" * 60)
    print("| Ngôn ngữ query | Tổng số Query | Số Hit | Recall@5 |")
    print("|:--------------:|:-------------:|:------:|:--------:|")
    print(f"| Tiếng Anh      | {en_results['total_queries']} | {en_results['hits']} | **{en_results['recall_at_5']:.4f}** |")
    print(f"| Tiếng Việt     | {vi_results['total_queries']} | {vi_results['hits']} | **{vi_results['recall_at_5']:.4f}** |")
    print("=" * 60)

    # Print 3-5 Hit Examples
    print("\n" + "=" * 60)
    print("CÁC VÍ DỤ QUERY TIẾNG VIỆT HIT ĐÚNG (CROSS-LINGUAL HOẠT ĐỘNG TỐT):")
    print("=" * 60)
    for i, item in enumerate(vi_hits[:5], 1):
        print(f"\n[{i}] Query: \"{item['query']}\"")
        print(f"    Expected: \"{item['expected_parent_title']}\"")
        print(f"    Hit at Rank: #{item['hit_rank']}")
        print("    Top-5 Results:")
        for r in item["top_5"]:
            mark = "-> [HIT]" if r["parent_title"].strip() == item["expected_parent_title"].strip() else "        "
            print(f"      {mark} #{r['rank']} (Score: {r['similarity']:.4f}) {r['parent_title']}")

    # Print 3-5 Miss Examples
    print("\n" + "=" * 60)
    print("CÁC VÍ DỤ QUERY TIẾNG VIỆT BỊ MISS (HẠN CHẾ CỦA CROSS-LINGUAL):")
    print("=" * 60)
    for i, item in enumerate(vi_misses[:5], 1):
        print(f"\n[{i}] Query: \"{item['query']}\"")
        print(f"    Expected: \"{item['expected_parent_title']}\"")
        print("    Top-5 Results:")
        for r in item["top_5"]:
            print(f"        #{r['rank']} (Score: {r['similarity']:.4f}) {r['parent_title']}")


def main():
    asyncio.run(run_experiment())


if __name__ == "__main__":
    main()
