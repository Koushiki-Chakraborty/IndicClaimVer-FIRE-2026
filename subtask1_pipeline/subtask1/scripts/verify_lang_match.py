import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1 import data_loader


def main():
    train, dev, pool = data_loader.load_all()
    if not train:
        print("[ERROR] train_subtask1.json not found -- place it in data/ first.")
        sys.exit(1)

    print(f"Loaded train set: {len(train)} items")

    mismatches = []
    lang_pair_counts = Counter()

    for item in train:
        claim_lang = data_loader.detect_lang(item["Text"])
        evidence_lang = data_loader.detect_lang(item["Evidence"])
        lang_pair_counts[(claim_lang, evidence_lang)] += 1
        if claim_lang != evidence_lang:
            mismatches.append({
                "id": item.get("ID"),
                "claim_lang": claim_lang,
                "evidence_lang": evidence_lang,
                "claim_snippet": item["Text"][:80],
                "evidence_snippet": item["Evidence"][:80],
            })

    print(f"\nTotal train items: {len(train)}")
    print(f"Mismatches (claim_lang != evidence_lang): {len(mismatches)}")
    print(f"\nAll (claim_lang, evidence_lang) pair counts:")
    for (cl, el), count in sorted(lang_pair_counts.items(), key=lambda x: -x[1]):
        marker = "" if cl == el else "  <-- MISMATCH"
        print(f"  claim={cl}  evidence={el}  n={count}{marker}")

    if mismatches:
        print(f"\nFirst 10 mismatches:")
        for m in mismatches[:10]:
            print(f"  ID={m['id']}  claim_lang={m['claim_lang']}  evidence_lang={m['evidence_lang']}")
            print(f"    Claim:    {m['claim_snippet']}")
            print(f"    Evidence: {m['evidence_snippet']}")

    # Verdict for docstring
    if len(mismatches) == 0:
        print("\nVERDICT: 0 mismatches confirmed. data_loader.py docstring is correct.")
    else:
        print(f"\nVERDICT: {len(mismatches)} mismatches found.")
        print("ACTION REQUIRED: Update the data_loader.py docstring to reflect")
        print(f"the true number ({len(mismatches)}/{len(train)} mismatches).")


if __name__ == "__main__":
    main()
