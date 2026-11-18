import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from subtask1 import data_loader
from subtask1.evaluate import evaluate_retrieval
from subtask1.neo4j_client import Neo4jClient
from kg_weighted_overlap_reranker import _entity_document_frequency_overlap

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "traindev_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
RESULTS_PATH = Path(__file__).parent / "weight_sweep_results.txt"

# Same weight candidates reported in weight_sweep_results.txt (see root README.md, Notes -> "KG weight sweep").
WEIGHTS_TO_TRY = [0.0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.2]


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
    gold_evidence_by_claim_id = {c["ID"]: c["Evidence"] for c in train + dev if "Evidence" in c}
    pool_text_to_id = data_loader.build_pool_text_to_id(pool)

    client = Neo4jClient()
    print(f"Target database: {client.database or '<server default>'} (read-only -- no writes to the graph)")

    # ---- One Neo4j pass: fetch dense score + weighted overlap for every claim's candidates ----
    per_claim = []  # (gold_text, [(evidence_id, dense_score, weighted_overlap)])
    for i, row in enumerate(dense_rows, start=1):
        candidates = [(e["ID"], e["score"]) for e in row["top10_evidence"]]
        candidate_ids = [c[0] for c in candidates]
        dense_scores = dict(candidates)
        weighted_overlap = _entity_document_frequency_overlap(client, row["ID"], candidate_ids)
        per_claim.append((
            gold_evidence_by_claim_id.get(row["ID"]),
            [(eid, dense_scores[eid], weighted_overlap.get(eid, 0.0)) for eid in candidate_ids],
        ))
        if i % 1000 == 0:
            print(f"  fetched {i}/{len(dense_rows)} claims...")
    client.close()

    # ---- Score every weight candidate against that same fetched data -- no more Neo4j calls ----
    header = f"{'Weight':>8} {'Success@3':>10} {'nDCG@3':>8}"
    print(f"\n{header}")
    summary_lines = [header]
    for weight in WEIGHTS_TO_TRY:
        retrieval_results = []
        for gold_text, candidates in per_claim:
            final_scores = {eid: dense + weight * overlap for eid, dense, overlap in candidates}
            top3_ids = sorted(final_scores, key=final_scores.get, reverse=True)[:3]
            if gold_text:
                retrieval_results.append({"claim_item": {"Evidence": gold_text}, "retrieved_ids": top3_ids})
        metrics = evaluate_retrieval(retrieval_results, pool_text_to_id, k=3)
        line = f"{weight:>8} {metrics['Success@3']:>10.4f} {metrics['nDCG@3']:>8.4f}"
        print(line)
        summary_lines.append(line)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nWrote sweep results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
