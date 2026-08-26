import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

CLAIM_FILES = {
    "bn": "subtask1_bengali_test.json",
    "hi": "subtask1_hindi_test.json",
    "en": "subtask1_english_test.json",
    "codemix": "subtask1_codemix_test.json",
}


def combine_claims():
    combined = []
    for lang, fname in CLAIM_FILES.items():
        path = DATA_DIR / fname
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            combined.append({"ID": item["ID"], "Text": item["Text"], "Lang": lang})
        print(f"  {fname}: {len(items)} claims (tagged Lang={lang!r})")

    ids = [c["ID"] for c in combined]
    assert len(ids) == len(set(ids)), "Duplicate claim IDs across the 4 source files!"

    out_path = DATA_DIR / "subtask1_all_test_combined.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(combined)} combined claims -> {out_path}")
    return combined


def fix_evidence_pool():
    src_path = DATA_DIR / "evidence_pool_test.json"
    with open(src_path, encoding="utf-8") as f:
        pool = json.load(f)

    fixed = [{"ID": item["ID"], "Evidence": item["evidence"]} for item in pool]

    out_path = DATA_DIR / "evidence_pool_test_fixed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(fixed)} evidence items (key renamed evidence -> Evidence) -> {out_path}")


if __name__ == "__main__":
    print("Combining per-language claim files...")
    combine_claims()
    print()
    print("Fixing evidence pool key naming...")
    fix_evidence_pool()
