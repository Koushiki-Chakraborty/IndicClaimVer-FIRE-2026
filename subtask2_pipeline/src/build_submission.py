"""
Collects every claim folder's verification_output.json (produced by
src/verification.py) into ONE final submission file: a UTF-8 JSON
array, one object per claim, in the exact FIRE 2026 Subtask-2 format:

    [
      {"ID": "...", "Evidence": "...", "Prediction": "SUPPORTS", "Justification": "..."},
      ...
    ]

Each per-claim verification_output.json is already in this exact
shape, so this script does no reformatting -- it only aggregates.

Usage:
    python -m scripts.build_submission
    python -m scripts.build_submission --limit 50
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config


def find_verification_records(claims_dir: Path) -> list:
    """Reads every claim folder's verification_output.json, in claim-ID
    sorted order (stable, reproducible submission file ordering)."""
    records = []
    missing = []
    for folder in sorted(claims_dir.iterdir()):
        if not folder.is_dir():
            continue
        record_path = folder / config.VERIFICATION_OUTPUT_FILENAME
        if not record_path.exists():
            missing.append(folder.name)
            continue
        with open(record_path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records, missing


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate every claim's verification_output.json into one final FIRE 2026 submission file."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Include at most this many records. Omit to include everything found.")
    # OPTIONAL ADD-ON: both default to the exact paths already used today,
    # so calling this script with no new flags is byte-for-byte identical
    # to before -- these only matter when explicitly overridden (e.g. by
    # run_pipeline.py's new --dataset-file / --output-file flags for a
    # separate dev-set run).
    parser.add_argument("--claims-dir", type=str, default=None,
                        help="Directory of per-claim folders to aggregate. "
                             "Default: config.CLAIMS_OUTPUT_DIR (unchanged existing behavior).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the aggregated submission JSON. "
                             "Default: config.OUTPUT_DIR / config.FINAL_SUBMISSION_FILENAME "
                             "(unchanged existing behavior).")
    args = parser.parse_args()

    claims_dir = Path(args.claims_dir) if args.claims_dir else config.CLAIMS_OUTPUT_DIR
    output_path = Path(args.output) if args.output else config.OUTPUT_DIR / config.FINAL_SUBMISSION_FILENAME

    records, missing = find_verification_records(claims_dir)

    if missing:
        print(f"[build_submission] WARNING: {len(missing)} claim folder(s) have no "
              f"verification_output.json yet, skipped: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")

    if args.limit is not None:
        records = records[:args.limit]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[build_submission] Wrote {len(records)} record(s) -> {output_path}")

    # Quick sanity check before you submit: every record must have a
    # valid, exact Prediction label.
    invalid = [r["ID"] for r in records if r.get("Prediction") not in config.VALID_PREDICTIONS]
    if invalid:
        print(f"[build_submission] WARNING: {len(invalid)} record(s) have an invalid "
              f"Prediction label and will likely be rejected by the FIRE submission "
              f"checker: {invalid}")


if __name__ == "__main__":
    main()