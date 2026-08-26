import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from subtask1.dense_gpt_classifier import run_dense_gpt_pipeline

HERE = Path(__file__).parent.parent  # officialTestData/
DATA_DIR = HERE / "data"
SUBTASK1_DIR = Path(__file__).parent.parent.parent  # subtask1/
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "cache" / "officialtest_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
GPT_CACHE_PATH = CACHE_DIR / "gpt_classification_cache.json"  # regenerable, stays in gitignored cache/
RESULTS_DIR = SUBTASK1_DIR / "results" / "testResult" / "dense_gpt"


def main(limit: int = None, max_workers: int = 10):
    if not DENSE_TOP10_PATH.exists():
        raise RuntimeError(
            f"{DENSE_TOP10_PATH} not found -- run "
            f"`python officialTestData/scripts/kg_pipeline.py dense` first."
        )
    with open(DENSE_TOP10_PATH, encoding="utf-8") as f:
        dense_rows = json.load(f)
    if limit:
        dense_rows = dense_rows[:limit]
    print(f"Loaded {len(dense_rows)} claims from {DENSE_TOP10_PATH}")

    with open(DATA_DIR / "evidence_pool_test_fixed.json", encoding="utf-8") as f:
        pool = json.load(f)
    pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
    pool_text_to_id = {p["Evidence"]: p["ID"] for p in pool}

    run_dense_gpt_pipeline(
        dense_rows=dense_rows,
        pool_id_to_text=pool_id_to_text,
        pool_text_to_id=pool_text_to_id,
        gold_evidence_by_id={},  # official test set has no gold evidence at all
        gold_label_by_id={},     # official test set has no gold labels at all
        cache_path=GPT_CACHE_PATH,
        results_dir=RESULTS_DIR,
        dataset_label="officialTestData",
        max_workers=max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N claims (for a quick test)")
    parser.add_argument("--max-workers", type=int, default=10, help="concurrent GPT-4o-mini requests")
    args = parser.parse_args()
    main(limit=args.limit, max_workers=args.max_workers)
