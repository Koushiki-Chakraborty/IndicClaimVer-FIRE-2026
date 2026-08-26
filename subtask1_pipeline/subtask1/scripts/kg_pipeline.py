import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "traindev_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
EVIDENCE_UNION_PATH = CACHE_DIR / "evidence_union.json"

# Final, tracked output -- NOT under cache/ (cache/* is gitignored, so
# anything written only there would never actually be committed/shipped).
# Own subfolder (kg/), same convention as results/dense/, results/nli_rerank/,
# results/bge_m3/, results/indic_sbert/ -- every retrieval approach gets
# its own results folder so they stay comparable.
RESULTS_DIR = Path(__file__).parent.parent / "results" / "devResult" / "kg"
# Separate folder for the +GPT output, same split as results/dense/ vs
# results/dense_gpt/ -- retrieval-only and retrieval+prediction outputs
# don't share a folder anywhere else in this project, KG shouldn't be
# the exception.
RESULTS_GPT_DIR = Path(__file__).parent.parent / "results" / "devResult" / "kg_gpt"

# detect_lang() only ever returns bn/hi/en (no separate code-mixed
# bucket) -- a code-mixed claim here just falls into whichever script
# dominates it, same as everywhere else in this project.
LANG_FILENAMES = {"bn": "bengali_result.json", "hi": "hindi_result.json", "en": "english_result.json"}


# ---------------------------------------------------------------------
# Stage 1: dense
# ---------------------------------------------------------------------
def stage_dense():
    import config
    from subtask1 import data_loader
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.retriever import DenseRetriever

    train, dev, pool = data_loader.load_all()
    all_claims = train + dev
    pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
    print(f"Loaded {len(all_claims)} claims ({len(train)} train + {len(dev)} dev), {len(pool)} evidence.")

    client = Neo4jClient()
    print(f"Target database: {client.database or '<server default>'}")

    n_with_emb = client.run("MATCH (e:Evidence) WHERE e.embedding IS NOT NULL RETURN count(e) AS n")[0]["n"]
    if n_with_emb == 0:
        raise RuntimeError(
            "No Evidence embeddings found in the target database -- Stage 1 "
            "(subtask1/scripts/03_build_kg_st1.py) needs to have run first."
        )

    dense = DenseRetriever(client)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dense_rows = []
    union_ids = set()

    for i, c in enumerate(all_claims, start=1):
        top10 = dense.retrieve(c["Text"], top_k=config.TOP_K_RETRIEVE, return_scores=True)
        dense_rows.append({
            "ID": c["ID"],
            "Claim": c["Text"],
            "top10_evidence": [
                {"ID": eid, "Evidence": pool_id_to_text.get(eid, ""), "score": score}
                for eid, score in top10
            ],
        })
        union_ids.update(eid for eid, _ in top10)
        if i % 500 == 0:
            print(f"  {i}/{len(all_claims)} claims retrieved, union size so far: {len(union_ids)}")

    with open(DENSE_TOP10_PATH, "w", encoding="utf-8") as f:
        json.dump(dense_rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(dense_rows)} rows -> {DENSE_TOP10_PATH}")

    with open(EVIDENCE_UNION_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(union_ids), f, ensure_ascii=False, indent=2)
    print(f"Wrote evidence_union.json: {len(union_ids)}/{len(pool)} evidence items "
          f"({100 * len(union_ids) / len(pool):.1f}% of the full pool) -> {EVIDENCE_UNION_PATH}")

    client.close()


# ---------------------------------------------------------------------
# Stage 2: shard
# ---------------------------------------------------------------------
def stage_shard():
    shard_index = int(os.environ["SHARD_INDEX"])
    shard_count = int(os.environ["SHARD_COUNT"])
    assert 0 <= shard_index < shard_count, "SHARD_INDEX must be in [0, SHARD_COUNT)"

    # MUST happen before importing subtask1.integrated_pipeline -- its cache
    # path constants are computed once at module-import time.
    os.environ["CACHE_SHARD_SUFFIX"] = f"shard{shard_index}"

    from subtask1 import data_loader
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.kg_builder_st1 import ingest_entities_and_mentions

    def my_slice(items: list) -> list:
        return [item for i, item in enumerate(items) if i % shard_count == shard_index]

    def load_evidence_union(pool: list) -> list:
        if not EVIDENCE_UNION_PATH.exists():
            raise RuntimeError(
                f"{EVIDENCE_UNION_PATH} not found -- run "
                f"`python subtask1/scripts/kg_pipeline.py dense` first "
                f"(needs Stage 1/2 done too) to compute which evidence actually needs linking."
            )
        with open(EVIDENCE_UNION_PATH, encoding="utf-8") as f:
            union_ids = set(json.load(f))
        reduced_pool = [p for p in pool if p["ID"] in union_ids]
        print(f"[shard {shard_index}/{shard_count}] Evidence union: {len(reduced_pool)}/{len(pool)} "
              f"items actually need linking ({100 * len(reduced_pool) / len(pool):.1f}% of the full pool).")
        return reduced_pool

    train, dev, pool = data_loader.load_all()
    if not pool:
        print("[shard] No pool data found -- place evidence_pool.json in the data/ directory first.")
        return

    client = Neo4jClient()
    client.setup_schema()  # idempotent, safe to call from every shard

    reduced_pool = load_evidence_union(pool)
    my_pool = my_slice(reduced_pool)
    print(f"[shard {shard_index}/{shard_count}] Evidence: {len(my_pool)}/{len(reduced_pool)} "
          f"(of the reduced union) assigned to this shard.")
    print(f"\n=== Stage 3 (shard {shard_index}): Entity extraction + linking (Evidence, reduced set) ===")
    ingest_entities_and_mentions(
        client, my_pool, id_field="ID", text_field="Evidence", node_label="Evidence"
    )

    all_claims = train + dev
    if all_claims:
        my_claims = my_slice(all_claims)
        print(f"[shard {shard_index}/{shard_count}] Claims: {len(my_claims)}/{len(all_claims)} items assigned to this shard.")
        print(f"\n=== Stage 3b (shard {shard_index}): Entity extraction + linking (Claims) ===")
        from subtask1 import embeddings
        embeddings.prepare_pool_embeddings(pool)  # no-op if already cached from Stage 1
        pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
        ingest_entities_and_mentions(
            client, my_claims, id_field="ID", text_field="Text", node_label="Claim",
            enable_llm_tier2=True, pool_id_to_text=pool_id_to_text,
        )

    print(f"\n[shard {shard_index}/{shard_count}] Done.")
    client.close()


# ---------------------------------------------------------------------
# Stage 3: cluster
# ---------------------------------------------------------------------
def stage_cluster():
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.event_clustering import build_event_clusters

    client = Neo4jClient()
    print(f"Target database: {client.database or '<server default>'}")
    n_evidence = client.run("MATCH (e:Evidence) RETURN count(e) AS n")[0]["n"]
    if n_evidence == 0:
        raise RuntimeError(
            "No Evidence nodes found -- Stage 1/2 (subtask1/scripts/03_build_kg_st1.py) "
            "needs to have run first."
        )
    build_event_clusters(client)
    client.close()


# ---------------------------------------------------------------------
# Stage 4: finalize
# ---------------------------------------------------------------------
def stage_finalize(limit: int = None, with_gpt: bool = False):
    from subtask1 import data_loader
    from subtask1.evaluate import evaluate_retrieval
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.retriever import DenseRetriever, KGBoostedRetriever

    if not DENSE_TOP10_PATH.exists():
        raise RuntimeError(
            f"{DENSE_TOP10_PATH} not found -- run "
            f"`python subtask1/scripts/kg_pipeline.py dense` first."
        )
    with open(DENSE_TOP10_PATH, encoding="utf-8") as f:
        dense_rows = json.load(f)
    if limit:
        dense_rows = dense_rows[:limit]

    train, dev, pool = data_loader.load_all()
    gold_evidence_by_claim_id = {c["ID"]: c["Evidence"] for c in train + dev if "Evidence" in c}

    client = Neo4jClient()
    print(f"Target database: {client.database or '<server default>'}")
    kg = KGBoostedRetriever(client, dense=None)  # score_precomputed() never touches .dense

    output_rows_by_lang = {"bn": [], "hi": [], "en": []}
    all_output_rows = []  # same rows as output_rows_by_lang, one flat list, claim order preserved -- for the classifier
    kg_reranked_rows = []  # flat, unsplit -- feeds the optional GPT step below
    retrieval_results = []
    for row in dense_rows:
        candidates = [(e["ID"], e["score"]) for e in row["top10_evidence"]]
        id_to_text = {e["ID"]: e["Evidence"] for e in row["top10_evidence"]}
        dense_score_by_id = dict(candidates)
        top3_ids = kg.score_precomputed(row["ID"], candidates, top_k=3)
        if len(top3_ids) < 3:
            padding = [eid for eid, _ in candidates if eid not in top3_ids]
            top3_ids = (top3_ids + padding)[:3]

        lang = data_loader.detect_lang(row["Claim"])
        output_row = {
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top_3_evidence": [
                {"ID": eid, "Evidence": id_to_text.get(eid, "")} for eid in top3_ids
            ],
        }
        output_rows_by_lang[lang].append(output_row)
        all_output_rows.append(output_row)

        # Same shape run_dense_gpt_pipeline() expects ("top10_evidence", even
        # though there are only 3 here, KG-reranked order) -- score is the
        # original Dense cosine similarity for that candidate, not a
        # recomputed KG score (score_precomputed() only returns ranked IDs,
        # not per-candidate scores) -- still a real, honest number, just not
        # the exact boosted value.
        kg_reranked_rows.append({
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top10_evidence": [
                {"ID": eid, "Evidence": id_to_text.get(eid, ""), "score": dense_score_by_id.get(eid, 0.0)}
                for eid in top3_ids
            ],
        })

        gold_text = gold_evidence_by_claim_id.get(row["ID"])
        if gold_text:
            retrieval_results.append({"claim_item": {"Evidence": gold_text}, "retrieved_ids": top3_ids})

    client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["=== KG-boosted retrieval (data/ train+dev) ==="]
    if retrieval_results:
        pool_text_to_id = data_loader.build_pool_text_to_id(pool)
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
    print(f"Wrote summary -> {RESULTS_DIR / 'summary.txt'}")

    for lang, fname in LANG_FILENAMES.items():
        out_path = RESULTS_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_rows_by_lang[lang], f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(output_rows_by_lang[lang])} rows -> {out_path}")

    # Combined, all-languages file -- same rows as the 3 per-language files
    # above, just merged into one list (claim order preserved) so the
    # classifier can read a single file instead of 3.
    combined_path = RESULTS_DIR / "result.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_output_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_output_rows)} rows -> {combined_path}")

    if with_gpt:
        from subtask1.dense_gpt_classifier import run_dense_gpt_pipeline

        gold_label_by_claim_id = {c["ID"]: c["Label"] for c in train + dev if "Label" in c}
        pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
        pool_text_to_id = data_loader.build_pool_text_to_id(pool)
        print(f"\n=== GPT-4o-mini veracity classification on the KG-boosted top-3 ===")
        run_dense_gpt_pipeline(
            dense_rows=kg_reranked_rows,
            pool_id_to_text=pool_id_to_text,
            pool_text_to_id=pool_text_to_id,
            gold_evidence_by_id=gold_evidence_by_claim_id,
            gold_label_by_id=gold_label_by_claim_id,
            cache_path=CACHE_DIR / "kg_gpt_classification_cache.json",
            results_dir=RESULTS_GPT_DIR,
            dataset_label="data/ train+dev",
            output_prefix="kg_gpt",
            retrieval_label="KG-boosted",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("dense")
    sub.add_parser("shard")
    sub.add_parser("cluster")
    p_finalize = sub.add_parser("finalize")
    p_finalize.add_argument("--limit", type=int, default=None,
                             help="only finalize the first N claims (for a quick test)")
    p_finalize.add_argument("--with-gpt", action="store_true",
                             help="also run GPT-4o-mini veracity classification on the KG-boosted "
                                  "top-3 and write the full official-schema submission "
                                  "(ID/top_3_evidence/Prediction) -- real OpenAI API calls, off by default")
    args = parser.parse_args()

    if args.stage == "dense":
        stage_dense()
    elif args.stage == "shard":
        stage_shard()
    elif args.stage == "cluster":
        stage_cluster()
    elif args.stage == "finalize":
        stage_finalize(limit=args.limit, with_gpt=args.with_gpt)


if __name__ == "__main__":
    main()
