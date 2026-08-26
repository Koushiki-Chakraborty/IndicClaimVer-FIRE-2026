import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

HERE = Path(__file__).parent.parent  # officialTestData/
DATA_DIR = HERE / "data"
SUBTASK1_DIR = Path(__file__).parent.parent.parent  # subtask1/

# Intermediate, regenerable (just rerun `dense`) -- lives under the repo's
# gitignored cache/, NOT results/, since it's large (tens of MB) and not
# a deliverable.
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "cache" / "officialtest_results"
DENSE_TOP10_PATH = CACHE_DIR / "dense_top10.json"
EVIDENCE_UNION_PATH = CACHE_DIR / "evidence_union.json"

# Final, tracked output -- the 4 per-language files only. Written into
# subtask1/results/testResult/ (the official-test-set counterpart of
# subtask1/results/devResult/), NOT officialTestData/results/ -- same
# convention as results/dense/ under subtask1/results/devResult/.
RESULTS_DIR = SUBTASK1_DIR / "results" / "testResult" / "kg"
# Separate folder for the +GPT official-schema submission -- same split
# as results/dense/ vs results/dense_gpt/, KG shouldn't be the exception.
RESULTS_GPT_DIR = SUBTASK1_DIR / "results" / "testResult" / "kg_gpt"

# Same 4-way split the organizers gave us (subtask1_bengali_test.json,
# subtask1_hindi_test.json, subtask1_english_test.json, subtask1_codemix_test.json).
LANG_FILENAMES = {
    "bn": "bengali_result.json", "hi": "hindi_result.json",
    "en": "english_result.json", "codemix": "codemix_result.json",
}


def load_test_data():
    with open(DATA_DIR / "subtask1_all_test_combined.json", encoding="utf-8") as f:
        claims = json.load(f)
    with open(DATA_DIR / "evidence_pool_test_fixed.json", encoding="utf-8") as f:
        pool = json.load(f)
    return claims, pool


def _warn_if_not_testkg(client):
    if client.database != "testkg":
        print(f"WARNING: NEO4J_DATABASE is {client.database!r}, not 'testkg' -- "
              f"set NEO4J_DATABASE=testkg before running this if that wasn't intended.")


# ---------------------------------------------------------------------
# Stage 1: setup
# ---------------------------------------------------------------------
def stage_setup():
    from subtask1 import data_loader, embeddings
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.kg_builder_st1 import ingest_evidence_nodes, _batched, BATCH_SIZE

    claims, pool = load_test_data()
    print(f"Loaded {len(claims)} claims, {len(pool)} evidence.")

    client = Neo4jClient()
    print(f"Target database: {client.database}")
    _warn_if_not_testkg(client)
    client.setup_schema()

    print("\n=== Stage 1: Evidence nodes + embeddings ===")
    ingest_evidence_nodes(client, pool)

    print("\n=== Stage 2: Claim nodes + embeddings (split=test) ===")
    rows = [{"id": c["ID"], "text": c["Text"], "lang": data_loader.detect_lang(c["Text"]), "split": "test"}
            for c in claims]
    vecs = embeddings.encode_batch([r["text"] for r in rows])
    for r, v in zip(rows, vecs):
        r["embedding"] = v
    for batch in _batched(rows, BATCH_SIZE):
        client.run_write(
            """
            UNWIND $rows AS row
            MERGE (c:Claim {id: row.id})
            SET c.text = row.text, c.lang = row.lang, c.split = row.split, c.embedding = row.embedding
            """,
            rows=batch,
        )

    stats = client.run("""
        OPTIONAL MATCH (e:Evidence) WITH count(e) AS n_evidence
        OPTIONAL MATCH (c:Claim) RETURN n_evidence, count(c) AS n_claim
    """)
    print("\nDone. Stats:", dict(stats[0]) if stats else "empty")
    client.close()


# ---------------------------------------------------------------------
# Stage 2: dense
# ---------------------------------------------------------------------
def stage_dense():
    import config
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.retriever import DenseRetriever

    claims, pool = load_test_data()
    pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
    print(f"Loaded {len(claims)} claims, {len(pool)} evidence.")

    client = Neo4jClient()
    print(f"Target database: {client.database}")
    _warn_if_not_testkg(client)

    n_with_emb = client.run("MATCH (e:Evidence) WHERE e.embedding IS NOT NULL RETURN count(e) AS n")[0]["n"]
    if n_with_emb == 0:
        raise RuntimeError(
            "No Evidence embeddings found -- run "
            "`python officialTestData/scripts/kg_pipeline.py setup` first (Stage 1/2)."
        )

    dense = DenseRetriever(client)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dense_rows = []
    union_ids = set()

    for i, c in enumerate(claims, start=1):
        top10 = dense.retrieve(c["Text"], top_k=config.TOP_K_RETRIEVE, return_scores=True)
        dense_rows.append({
            "ID": c["ID"],
            "Claim": c["Text"],
            "Lang": c["Lang"],
            "top10_evidence": [
                {"ID": eid, "Evidence": pool_id_to_text.get(eid, ""), "score": score}
                for eid, score in top10
            ],
        })
        union_ids.update(eid for eid, _ in top10)
        if i % 500 == 0:
            print(f"  {i}/{len(claims)} claims retrieved, union size so far: {len(union_ids)}")

    with open(DENSE_TOP10_PATH, "w", encoding="utf-8") as f:
        json.dump(dense_rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(dense_rows)} rows -> {DENSE_TOP10_PATH}")

    with open(EVIDENCE_UNION_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(union_ids), f, ensure_ascii=False, indent=2)
    print(f"Wrote evidence_union.json: {len(union_ids)}/{len(pool)} evidence items "
          f"({100 * len(union_ids) / len(pool):.1f}% of the full pool) -> {EVIDENCE_UNION_PATH}")

    client.close()


# ---------------------------------------------------------------------
# Stage 3: shard
# ---------------------------------------------------------------------
def stage_shard():
    shard_index = int(os.environ["SHARD_INDEX"])
    shard_count = int(os.environ["SHARD_COUNT"])
    assert 0 <= shard_index < shard_count, "SHARD_INDEX must be in [0, SHARD_COUNT)"

    os.environ["CACHE_SHARD_SUFFIX"] = f"testshard{shard_index}"  # before importing integrated_pipeline

    from subtask1 import embeddings
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.kg_builder_st1 import ingest_entities_and_mentions

    def my_slice(items: list) -> list:
        return [item for i, item in enumerate(items) if i % shard_count == shard_index]

    def load_evidence_union(pool: list) -> list:
        if not EVIDENCE_UNION_PATH.exists():
            raise RuntimeError(
                f"{EVIDENCE_UNION_PATH} not found -- run "
                f"`python officialTestData/scripts/kg_pipeline.py dense` first "
                f"(needs `setup` done too) to compute which evidence actually needs linking."
            )
        with open(EVIDENCE_UNION_PATH, encoding="utf-8") as f:
            union_ids = set(json.load(f))
        reduced_pool = [p for p in pool if p["ID"] in union_ids]
        print(f"[test-shard {shard_index}/{shard_count}] Evidence union: {len(reduced_pool)}/{len(pool)} "
              f"items actually need linking ({100 * len(reduced_pool) / len(pool):.1f}% of the full pool).")
        return reduced_pool

    claims, pool = load_test_data()
    print(f"[test-shard {shard_index}/{shard_count}] Loaded {len(claims)} claims, {len(pool)} evidence "
          f"(full pool, before reducing to the union).")

    client = Neo4jClient()
    print(f"[test-shard {shard_index}/{shard_count}] Target database: {client.database}")
    client.setup_schema()  # idempotent, safe from every shard

    n_evidence = client.run("MATCH (e:Evidence) RETURN count(e) AS n")[0]["n"]
    if n_evidence == 0:
        raise RuntimeError(
            "No Evidence nodes found -- run "
            "`python officialTestData/scripts/kg_pipeline.py setup` once first (Stage 1/2), "
            "before launching shards."
        )
    print(f"[test-shard {shard_index}/{shard_count}] Found {n_evidence} Evidence nodes already present -- "
          f"Stage 1/2 already done, proceeding to Stage 3.")

    reduced_pool = load_evidence_union(pool)
    my_pool = my_slice(reduced_pool)
    print(f"[test-shard {shard_index}/{shard_count}] Evidence: {len(my_pool)}/{len(reduced_pool)} "
          f"(of the reduced union) assigned to this shard.")
    print(f"\n=== Stage 3 (shard {shard_index}): Entity extraction + linking (Evidence, reduced set) ===")
    ingest_entities_and_mentions(
        client, my_pool, id_field="ID", text_field="Evidence", node_label="Evidence"
    )

    my_claims = my_slice(claims)
    print(f"[test-shard {shard_index}/{shard_count}] Claims: {len(my_claims)}/{len(claims)} assigned to this shard.")
    print(f"\n=== Stage 3b (shard {shard_index}): Entity extraction + linking (Claims) ===")
    embeddings.prepare_pool_embeddings(pool)
    pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
    ingest_entities_and_mentions(
        client, my_claims, id_field="ID", text_field="Text", node_label="Claim",
        enable_llm_tier2=True, pool_id_to_text=pool_id_to_text,
    )

    print(f"\n[test-shard {shard_index}/{shard_count}] Done.")
    client.close()


# ---------------------------------------------------------------------
# Stage 4: cluster
# ---------------------------------------------------------------------
def stage_cluster():
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.event_clustering import build_event_clusters

    client = Neo4jClient()
    print(f"Target database: {client.database}")
    _warn_if_not_testkg(client)
    n_evidence = client.run("MATCH (e:Evidence) RETURN count(e) AS n")[0]["n"]
    if n_evidence == 0:
        raise RuntimeError(
            "No Evidence nodes found -- run "
            "`python officialTestData/scripts/kg_pipeline.py setup` first."
        )
    build_event_clusters(client)
    client.close()


# ---------------------------------------------------------------------
# Stage 5: finalize
# ---------------------------------------------------------------------
def stage_finalize(limit: int = None, with_gpt: bool = False):
    from subtask1.neo4j_client import Neo4jClient
    from subtask1.retriever import KGBoostedRetriever

    if not DENSE_TOP10_PATH.exists():
        raise RuntimeError(
            f"{DENSE_TOP10_PATH} not found -- run "
            f"`python officialTestData/scripts/kg_pipeline.py dense` first."
        )
    with open(DENSE_TOP10_PATH, encoding="utf-8") as f:
        dense_rows = json.load(f)
    if limit:
        dense_rows = dense_rows[:limit]

    # Look up Lang from the organizer-provided claims data, not the cached
    # dense_top10.json row -- older dense_top10.json files (generated
    # before "Lang" was added to stage_dense's output) don't have this key,
    # and re-deriving it from claims here means finalize keeps working
    # against a stale cache instead of a bare KeyError.
    claims, _pool = load_test_data()
    id_to_lang = {c["ID"]: c["Lang"] for c in claims}

    client = Neo4jClient()
    print(f"Target database: {client.database}")
    _warn_if_not_testkg(client)
    kg = KGBoostedRetriever(client, dense=None)  # score_precomputed() never touches .dense

    output_rows_by_lang = {"bn": [], "hi": [], "en": [], "codemix": []}
    kg_reranked_rows = []  # flat, unsplit -- feeds the optional GPT step below
    for row in dense_rows:
        candidates = [(e["ID"], e["score"]) for e in row["top10_evidence"]]
        id_to_text = {e["ID"]: e["Evidence"] for e in row["top10_evidence"]}
        dense_score_by_id = dict(candidates)
        top3_ids = kg.score_precomputed(row["ID"], candidates, top_k=3)
        if len(top3_ids) < 3:
            padding = [eid for eid, _ in candidates if eid not in top3_ids]
            top3_ids = (top3_ids + padding)[:3]

        output_rows_by_lang[id_to_lang[row["ID"]]].append({
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top_3_evidence": [
                {"ID": eid, "Evidence": id_to_text.get(eid, "")} for eid in top3_ids
            ],
        })

        # Same shape run_dense_gpt_pipeline() expects -- score is the
        # original Dense cosine similarity for that candidate, not a
        # recomputed KG score (score_precomputed() only returns ranked
        # IDs, not per-candidate scores).
        kg_reranked_rows.append({
            "ID": row["ID"],
            "Claim": row["Claim"],
            "top10_evidence": [
                {"ID": eid, "Evidence": id_to_text.get(eid, ""), "score": dense_score_by_id.get(eid, 0.0)}
                for eid in top3_ids
            ],
        })

    client.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nNo gold evidence exists for the official test set -- no Success@3/nDCG@3 to report here.")
    for lang, fname in LANG_FILENAMES.items():
        out_path = RESULTS_DIR / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_rows_by_lang[lang], f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(output_rows_by_lang[lang])} rows -> {out_path}")

    if with_gpt:
        from subtask1.dense_gpt_classifier import run_dense_gpt_pipeline

        pool_id_to_text = {e["ID"]: e["Evidence"] for row in dense_rows for e in row["top10_evidence"]}
        pool_text_to_id = {v: k for k, v in pool_id_to_text.items()}
        print(f"\n=== GPT-4o-mini veracity classification on the KG-boosted top-3 ===")
        run_dense_gpt_pipeline(
            dense_rows=kg_reranked_rows,
            pool_id_to_text=pool_id_to_text,
            pool_text_to_id=pool_text_to_id,
            gold_evidence_by_id={},  # official test set has no gold evidence at all
            gold_label_by_id={},     # official test set has no gold labels at all
            cache_path=CACHE_DIR / "kg_gpt_classification_cache.json",
            results_dir=RESULTS_GPT_DIR,
            dataset_label="officialTestData",
            output_prefix="kg_gpt",
            retrieval_label="KG-boosted",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("setup")
    sub.add_parser("dense")
    sub.add_parser("shard")
    sub.add_parser("cluster")
    p_finalize = sub.add_parser("finalize")
    p_finalize.add_argument("--limit", type=int, default=None,
                             help="only finalize the first N claims (for a quick test)")
    p_finalize.add_argument("--with-gpt", action="store_true",
                             help="also run GPT-4o-mini veracity classification on the KG-boosted "
                                  "top-3 and write the full official-schema submission -- real "
                                  "OpenAI API calls, off by default")
    args = parser.parse_args()

    if args.stage == "setup":
        stage_setup()
    elif args.stage == "dense":
        stage_dense()
    elif args.stage == "shard":
        stage_shard()
    elif args.stage == "cluster":
        stage_cluster()
    elif args.stage == "finalize":
        stage_finalize(limit=args.limit, with_gpt=args.with_gpt)


if __name__ == "__main__":
    main()
