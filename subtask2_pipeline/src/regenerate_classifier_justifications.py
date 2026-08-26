"""
Replaces the templated Justification field in every existing
classifier_output record with a real, GPT-4o-mini-generated
justification for the ALREADY-DECIDED MuRIL prediction (see
src/gpt_justifier.py -- this never reclassifies, only explains).

Reads and OVERWRITES, in place, the exact files
src/run_classifier_pipeline.py already produced:
    output/classifier_output/dev_submission.json
    output/classifier_output/language_outputs/{English,Hindi,Bengali,CodeMix}_subtask2.json
Same schema, same IDs, same Prediction values, same Evidence text --
ONLY the Justification field changes. dev_metrics.json is untouched
(Prediction values don't change, so accuracy/macro-F1 don't either).

The classifier_output records only carry {ID, Evidence, Prediction,
Justification} -- no claim text -- so claim text is recovered by ID
from data/dev_subtask2.json (for dev_submission.json) and the four
data/subtask2_*_test.json files (for language_outputs/*.json).

Usage:
    python -m src.regenerate_classifier_justifications
    python -m src.regenerate_classifier_justifications --limit 20   # smoke test
    python -m src.regenerate_classifier_justifications --max-workers 5
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.gpt_justifier import justify

CLASSIFIER_OUTPUT_DIR = config.OUTPUT_DIR / "classifier_output"

TARGET_FILES = [
    CLASSIFIER_OUTPUT_DIR / "dev_submission.json",
    CLASSIFIER_OUTPUT_DIR / "language_outputs" / "English_subtask2.json",
    CLASSIFIER_OUTPUT_DIR / "language_outputs" / "Hindi_subtask2.json",
    CLASSIFIER_OUTPUT_DIR / "language_outputs" / "Bengali_subtask2.json",
    CLASSIFIER_OUTPUT_DIR / "language_outputs" / "CodeMix_subtask2.json",
]


def _build_claim_text_lookup() -> dict:
    """{ID: Text} across dev_subtask2.json and all 4 test language files."""
    lookup = {}
    files = [config.DATA_DIR / "dev_subtask2.json"] + list(config.DATA_FILES.values())
    for path in files:
        if not path.exists():
            print(f"[regenerate_justifications] WARNING: {path} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            lookup[item["ID"]] = item["Text"]
    return lookup


def _process_file(path: Path, claim_text_by_id: dict, max_workers: int, limit: int = None) -> None:
    if not path.exists():
        print(f"[regenerate_justifications] WARNING: {path} not found, skipping.")
        return

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    # BUG FIX: `--limit` must only cap how many records get PROCESSED
    # (billed) for a quick smoke test -- it must NEVER shrink `records`
    # itself, since `records` (the full list) is what gets written back
    # to disk at the end. An earlier version of this function did
    # `records = records[:limit]`, which truncated the file's full
    # contents down to `limit` items on write -- a real data-loss bug,
    # caught after a --limit 2 smoke test wiped 629/631 dev records
    # (and similarly for the four test language files) down to 2 each.
    missing_claim_text = [r["ID"] for r in records if r["ID"] not in claim_text_by_id]
    if missing_claim_text:
        print(f"[regenerate_justifications] WARNING: {len(missing_claim_text)} record(s) in "
              f"{path.name} have no matching claim text, skipping those: {missing_claim_text[:5]}")
        records_to_process = [r for r in records if r["ID"] in claim_text_by_id]
    else:
        records_to_process = list(records)
    if limit is not None:
        records_to_process = records_to_process[:limit]

    print(f"[regenerate_justifications] {path.name}: justifying {len(records_to_process)} record(s)...")

    def _work(record):
        claim_text = claim_text_by_id[record["ID"]]
        new_justification = justify(claim_text, record["Evidence"], record["Prediction"])
        return record["ID"], new_justification

    updated_by_id = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_work, r): r for r in records_to_process}
        done = 0
        for future in as_completed(futures):
            claim_id, new_justification = future.result()
            updated_by_id[claim_id] = new_justification
            done += 1
            if done % 50 == 0 or done == len(records_to_process):
                print(f"  {done}/{len(records_to_process)} justified...")

    for r in records:
        if r["ID"] in updated_by_id:
            r["Justification"] = updated_by_id[r["ID"]]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[regenerate_justifications] Wrote {len(records)} record(s) -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N records per file (for a quick smoke test).")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Concurrent OpenAI requests per file (default: 10).")
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        print("[regenerate_justifications] ERROR: OPENAI_API_KEY not set in .env. Aborting.")
        sys.exit(1)

    claim_text_by_id = _build_claim_text_lookup()
    print(f"[regenerate_justifications] Loaded claim text for {len(claim_text_by_id)} claims.")

    for path in TARGET_FILES:
        _process_file(path, claim_text_by_id, max_workers=args.max_workers, limit=args.limit)

    print("\n[regenerate_justifications] Done.")
