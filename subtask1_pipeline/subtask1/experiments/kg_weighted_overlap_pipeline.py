import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from subtask1 import data_loader
from subtask1.evaluate import evaluate_retrieval
from subtask1.neo4j_client import Neo4jClient
from kg_weighted_overlap_reranker import score_precomputed_weighted_overlap

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "traindev_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "kg_test"


def main(limit: int = None):
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

    train, dev, pool = data_loader.load_all()
    gold_evidence_by_claim_id = {c["ID"]: c["Evidence"] for c in train + dev if "Evidence" in c}
    pool_text_to_id = data_loader.build_pool_text_to_id(pool)

    client = Neo4jClient()
    print(f"Target database: {client.database or '<server default>'} (read-only -- no writes to the graph)")

    output_rows = []
    retrieval_results = []
    for row in dense_rows:
        candidates = [(e["ID"], e["score"]) for e in row["top10_evidence"]]
        id_to_text = {e["ID"]: e["Evidence"] for e in row["top10_evidence"]}
        top3_ids = score_precomputed_weighted_overlap(client, row["ID"], candidates, top_k=3)
        if len(top3_ids) < 3:
            padding = [eid for eid, _ in candidates if eid not in top3_ids]
            top3_ids = (top3_ids + padding)[:3]

        output_rows.append({
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top_3_evidence": [
                {"ID": eid, "Evidence": id_to_text.get(eid, "")} for eid in top3_ids
            ],
        })

        gold_text = gold_evidence_by_claim_id.get(row["ID"])
        if gold_text:
            retrieval_results.append({"claim_item": {"Evidence": gold_text}, "retrieved_ids": top3_ids})

    client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["=== IDF-weighted KG-boosted retrieval (data/ train+dev) -- EXPERIMENTAL, separate from results/kg/ ==="]
    if retrieval_results:
        metrics = evaluate_retrieval(retrieval_results, pool_text_to_id, k=3)
        print(f"\n=== Retrieval metrics ({len(retrieval_results)}/{len(dense_rows)} claims with gold evidence) ===")
        print(metrics)
        summary_lines.append(
            f"n={metrics['n']}  Success@3={metrics['Success@3']:.4f}  nDCG@3={metrics['nDCG@3']:.4f}"
        )
    else:
        print("\nNo gold evidence found for any claim in this batch -- skipping Success@3/nDCG@3.")
        summary_lines.append("No gold evidence found for this batch -- Success@3/nDCG@3 not computed.")

    with open(RESULTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nWrote summary -> {RESULTS_DIR / 'summary.txt'}")

    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output_rows)} rows -> {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N claims (for a quick test)")
    args = parser.parse_args()
    main(limit=args.limit)
