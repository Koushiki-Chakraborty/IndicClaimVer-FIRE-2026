import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1 import data_loader
from subtask1.nli_reranker import rerank_top10
from subtask1.dense_gpt_classifier import run_dense_gpt_pipeline

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "traindev_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
NLI_RERANKED_CACHE_PATH = CACHE_DIR / "nli_reranked_top10.json"  # the rerank itself is slow -- cache it
GPT_CACHE_PATH = CACHE_DIR / "nli_rerank_gpt_cache.json"  # SEPARATE from dense_gpt's cache -- different top-1 candidates
RESULTS_DIR = Path(__file__).parent.parent / "results" / "devResult" / "nli_rerank"


def main(limit: int = None, max_workers: int = 10):
    if not DENSE_TOP10_PATH.exists():
        raise RuntimeError(
            f"{DENSE_TOP10_PATH} not found -- run "
            f"`python subtask1/scripts/kg_pipeline.py dense` first."
        )
    with open(DENSE_TOP10_PATH, encoding="utf-8") as f:
        dense_rows = json.load(f)
    if limit:
        dense_rows = dense_rows[:limit]
    print(f"Loaded {len(dense_rows)} claims from {DENSE_TOP10_PATH}")

    nli_rows = None
    if NLI_RERANKED_CACHE_PATH.exists():
        with open(NLI_RERANKED_CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        cached_by_id = {r["ID"]: r for r in cached}
        if all(row["ID"] in cached_by_id for row in dense_rows):
            print(f"Loaded NLI-reranked candidates from cache -> {NLI_RERANKED_CACHE_PATH}")
            nli_rows = [cached_by_id[row["ID"]] for row in dense_rows]

    if nli_rows is None:
        print("Running NLI reranking (slow -- tens of minutes on CPU for the full set)...")
        nli_rows = rerank_top10(dense_rows)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(NLI_RERANKED_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(nli_rows, f, ensure_ascii=False)
        print(f"Cached NLI-reranked candidates -> {NLI_RERANKED_CACHE_PATH}")

    train, dev, pool = data_loader.load_all()
    gold_evidence_by_id = {c["ID"]: c["Evidence"] for c in train + dev if "Evidence" in c}
    gold_label_by_id = {c["ID"]: c["Label"] for c in train + dev if "Label" in c}
    pool_text_to_id = data_loader.build_pool_text_to_id(pool)
    pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}

    run_dense_gpt_pipeline(
        dense_rows=nli_rows,
        pool_id_to_text=pool_id_to_text,
        pool_text_to_id=pool_text_to_id,
        gold_evidence_by_id=gold_evidence_by_id,
        gold_label_by_id=gold_label_by_id,
        cache_path=GPT_CACHE_PATH,
        results_dir=RESULTS_DIR,
        dataset_label="data/ train+dev, NLI-reranked",
        max_workers=max_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N claims (for a quick test)")
    parser.add_argument("--max-workers", type=int, default=10, help="concurrent GPT-4o-mini requests")
    args = parser.parse_args()
    main(limit=args.limit, max_workers=args.max_workers)
