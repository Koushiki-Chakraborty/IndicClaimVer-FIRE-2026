import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1 import data_loader
from subtask1.dense_retrieval_report import write_dense_only_report

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "traindev_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "devResult" / "dense"

BUCKET_LABELS = {"hi": "Hindi", "bn": "Bengali", "en": "English", "code-mixed": "Code-mixed"}


def main():
    if not DENSE_TOP10_PATH.exists():
        raise RuntimeError(
            f"{DENSE_TOP10_PATH} not found -- run "
            f"`python subtask1/scripts/kg_pipeline.py dense` first."
        )
    with open(DENSE_TOP10_PATH, encoding="utf-8") as f:
        dense_rows = json.load(f)
    print(f"Loaded {len(dense_rows)} claims from {DENSE_TOP10_PATH}")

    train, dev, pool = data_loader.load_all()
    gold_evidence_by_id = {c["ID"]: c["Evidence"] for c in train + dev if "Evidence" in c}
    pool_text_to_id = data_loader.build_pool_text_to_id(pool)

    retrieval_by_bucket = {"hi": [], "bn": [], "en": [], "code-mixed": []}
    for row in dense_rows:
        top3_ids = [e["ID"] for e in row["top10_evidence"][:3]]
        gold_text = gold_evidence_by_id.get(row["ID"])
        if gold_text:
            bucket = "code-mixed" if data_loader.is_code_mixed(row["Claim"]) else data_loader.detect_lang(row["Claim"])
            retrieval_by_bucket[bucket].append(
                {"claim_item": {"Evidence": gold_text}, "retrieved_ids": top3_ids}
            )

    write_dense_only_report(
        dense_rows=dense_rows,
        results_dir=RESULTS_DIR,
        dataset_label="data/ train+dev",
        pool_text_to_id=pool_text_to_id,
        retrieval_by_bucket=retrieval_by_bucket,
        bucket_display_labels=BUCKET_LABELS,
    )


if __name__ == "__main__":
    main()
