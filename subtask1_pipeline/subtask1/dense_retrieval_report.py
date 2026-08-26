import json
from pathlib import Path

from subtask1.evaluate import evaluate_retrieval


def write_dense_only_report(
    dense_rows: list,
    results_dir: Path,
    dataset_label: str,
    pool_text_to_id: dict = None,
    retrieval_by_bucket: dict = None,   # {bucket_label: [ {"claim_item":..., "retrieved_ids":...}, ... ]} or None
    bucket_display_labels: dict = None,  # {bucket_key: "Display Name"}, in the order to print
):
    """Writes results.json (ID, Claim, top_3_evidence -- no Prediction) and
    summary.txt (Success@3/nDCG@3, per-bucket if retrieval_by_bucket is
    given, otherwise just overall; honestly reports "not available" if no
    gold evidence exists for this dataset at all)."""
    output_rows = []
    for row in dense_rows:
        top3 = row["top10_evidence"][:3]
        output_rows.append({
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top_3_evidence": [
                {"ID": e["ID"], "Evidence": e["Evidence"], "score": e["score"]} for e in top3
            ],
        })

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output_rows)} rows (no Prediction field) -> {out_json}")

    summary_lines = [f"=== Dense-only top-3 retrieval ({dataset_label}) ==="]
    if retrieval_by_bucket:
        all_results = [r for bucket_results in retrieval_by_bucket.values() for r in bucket_results]
        if all_results:
            overall = evaluate_retrieval(all_results, pool_text_to_id, k=3)
            summary_lines.append(f"{'Language':<12} {'Success@3':<12} {'nDCG@3':<12} {'n':<8}")
            summary_lines.append("-" * 44)
            for key, label in bucket_display_labels.items():
                m = evaluate_retrieval(retrieval_by_bucket[key], pool_text_to_id, k=3)
                summary_lines.append(f"{label:<12} {m['Success@3']:<12.3f} {m['nDCG@3']:<12.3f} {m['n']:<8}")
            summary_lines.append(f"{'Overall':<12} {overall['Success@3']:<12.3f} {overall['nDCG@3']:<12.3f} {overall['n']:<8}")
        else:
            summary_lines.append("No gold evidence available for this dataset -- Success@3/nDCG@3 not computed.")
    else:
        summary_lines.append(f"n = {len(dense_rows)}")
        summary_lines.append("No gold evidence available for this dataset -- Success@3/nDCG@3 not computed.")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    out_summary = results_dir / "summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nWrote summary -> {out_summary}")
