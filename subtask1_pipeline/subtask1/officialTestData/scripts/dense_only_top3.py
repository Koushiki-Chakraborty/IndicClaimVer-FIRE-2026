import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from subtask1.dense_retrieval_report import write_dense_only_report

HERE = Path(__file__).parent.parent  # officialTestData/
SUBTASK1_DIR = Path(__file__).parent.parent.parent  # subtask1/
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "cache" / "officialtest_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
RESULTS_DIR = SUBTASK1_DIR / "results" / "testResult" / "dense"


def main():
    if not DENSE_TOP10_PATH.exists():
        raise RuntimeError(
            f"{DENSE_TOP10_PATH} not found -- run "
            f"`python officialTestData/scripts/kg_pipeline.py dense` first."
        )
    with open(DENSE_TOP10_PATH, encoding="utf-8") as f:
        dense_rows = json.load(f)
    print(f"Loaded {len(dense_rows)} claims from {DENSE_TOP10_PATH}")

    write_dense_only_report(
        dense_rows=dense_rows,
        results_dir=RESULTS_DIR,
        dataset_label="officialTestData",
    )


if __name__ == "__main__":
    main()
