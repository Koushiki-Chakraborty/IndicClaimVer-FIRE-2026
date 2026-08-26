from collections import Counter

from tqdm import tqdm

from subtask1 import data_loader, entity_linker
from subtask1.neo4j_client import Neo4jClient
from subtask1 import embeddings

BATCH_SIZE = 200

# How often ingest_entities_and_mentions() flushes to Neo4j during the
# main NER+linking loop -- deliberately much smaller than BATCH_SIZE
# above (which is fine as-is for the cheap, in-memory Evidence/Claim
# node writes). Each item here costs real, slow work (NER + Wikidata +
# occasional LLM calls, tens of seconds), so a crash before the next
# flush loses that many items' worth of work, redone from scratch on
# resume. 200 meant up to ~2+ hours lost per crash on this project's
# actual measured per-item rate -- too coarse for a machine that's been
# repeatedly killed by RAM pressure mid-shard. 10 bounds that loss to a
# few minutes, at the cost of more (still cheap) Neo4j write calls.
LINKING_FLUSH_EVERY = 10


def _batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_evidence_nodes(client: Neo4jClient, pool: list):
    print("[kg_builder_st1] Encoding pool evidence embeddings...")
    texts = [item["Evidence"] for item in pool]
    langs = [data_loader.detect_lang(t) for t in texts]
    vecs = embeddings.encode_batch(texts, is_query=False)

    rows = [
        {"id": item["ID"], "text": item["Evidence"], "lang": lang, "embedding": vec}
        for item, lang, vec in zip(pool, langs, vecs)
    ]
    print(f"[kg_builder_st1] Writing {len(rows)} Evidence nodes to Neo4j...")
    for batch in tqdm(list(_batched(rows, BATCH_SIZE)), desc="Evidence nodes"):
        client.run_write(
            """
            UNWIND $rows AS row
            MERGE (e:Evidence {id: row.id})
            SET e.text = row.text, e.lang = row.lang, e.embedding = row.embedding
            """,
            rows=batch,
        )


def ingest_claim_nodes(client: Neo4jClient, train: list, dev: list):
    # No Label / GOLD_EVIDENCE here -- both only exist for train/dev (never
    # available at real test time), and neither was ever queried back out by
    # any code: training reads Label straight from the JSON files (see
    # classifier.py), and Success@k evaluation cross-references gold
    # evidence directly against the source JSON. Removed as unused KG noise.
    rows = []
    for split_name, split_data in [("train", train), ("dev", dev)]:
        for item in split_data:
            rows.append({
                "id": item["ID"],
                "text": item["Text"],
                "lang": data_loader.detect_lang(item["Text"]),
                "split": split_name,
            })
    print("[kg_builder_st1] Encoding claim embeddings...")
    vecs = embeddings.encode_batch([r["text"] for r in rows])
    for r, v in zip(rows, vecs):
        r["embedding"] = v

    print(f"[kg_builder_st1] Writing {len(rows)} Claim nodes to Neo4j...")
    for batch in tqdm(list(_batched(rows, BATCH_SIZE)), desc="Claim nodes"):
        client.run_write(
            """
            UNWIND $rows AS row
            MERGE (c:Claim {id: row.id})
            SET c.text = row.text, c.lang = row.lang,
                c.split = row.split, c.embedding = row.embedding
            """,
            rows=batch,
        )


def ingest_entities_and_mentions(client: Neo4jClient, items: list, id_field: str,
                                  text_field: str, node_label: str,
                                  enable_llm_tier2: bool = False,
                                  pool_id_to_text: dict = None) -> Counter:
    """items: list of dicts with an id and text field.
    node_label: 'Evidence' or 'Claim' -- which node type HAS_MENTION points from.

    enable_llm_tier2 / pool_id_to_text: second-tier LLM escalation (see
    integrated_pipeline.resolve_document()'s enable_llm_tier2 docstring).
    Only ever passed True for the Claims pass (node_label='Claim') in
    main() below -- deliberately NOT for Evidence, since an Evidence
    node's entity resolution is meant to be reusable across whatever
    claims eventually retrieve it, not tied to one specific claim's
    framing (see design discussion: Claim -> its own related evidence is
    safe and generalizes to real inference-time claims; Evidence -> the
    claim(s) that happen to cite it as gold is not, since most pool items
    have no such claim at all and a few are cited by more than one).

    Uses the validated entity-linking pipeline (resolve_document(), from
    subtask1/integrated_pipeline.py): NER -> candidate generation with
    the Layer 2 cascade-control-flow fix (a stage's non-empty result no
    longer stops the search when the pool is weak) -> Layer 3 lexical
    ranking fix (compound-match preference is soft, not absolute) ->
    semantic reranking (context-sentence-vs-Wikidata-description cosine
    similarity, on top of the same candidate pool -- never generates new
    candidates) -> LLM fallback (enable_llm_fallback=True; gpt-4o-mini,
    last resort only, when the pool is structurally weak or the top-1
    doesn't clearly beat the runner-up -- see integrated_pipeline.py's
    resolve_document() docstring). Enabled here after a representative
    (not cherry-picked) evaluation across data_test/: 22.9% of mentions
    triggered it, 77% of those produced a genuine fix, 23% were harmless
    wasted calls, 0 confirmed regressions after fixing a type-flag cache
    -poisoning bug the evaluation itself surfaced. Neighbor-country
    reasoning stays OFF (confirmed P17-vs-P27 bug, unfixed) --
    resolve_document()'s own default handles that.

    Writes the finalized schema: src -[:HAS_MENTION]-> Mention
    -[:LINKS_TO]-> Entity {qid, label, description}, plus the
    src -[:MENTIONS]-> Entity shortcut. No Mention node existed in this
    function before -- mention text used to be a bare property on a
    direct src->Entity MENTIONS edge; that direct edge is now only the
    dedup/query-performance shortcut, not the primary record.

    Returns a Counter of qid -> mention count across this batch of items
    (available for stats/reporting).

    RESUMABLE: writes flush to Neo4j every LINKING_FLUSH_EVERY items (not
    once at the very end), and any item already carrying a HAS_MENTION edge from
    a previous run is skipped entirely -- so stopping this partway through
    (Ctrl+C, crash, end of the day) and re-running the same command later
    resumes from roughly where it left off instead of restarting item 0.
    An item with genuinely zero mentions extracted never gets a HAS_MENTION
    edge either way, so it's harmlessly re-run (cheap local NER only, no
    network cost) rather than skipped -- not worth the extra bookkeeping
    to special-case.
    """
    from subtask1.integrated_pipeline import resolve_document

    already_done = {r["id"] for r in client.run(
        f"MATCH (src:{node_label})-[:HAS_MENTION]->() RETURN DISTINCT src.id AS id"
    )}
    todo = [item for item in items if item[id_field] not in already_done]
    if already_done:
        print(f"[kg_builder_st1] Resuming ({node_label}): {len(already_done)} already linked, "
              f"{len(todo)}/{len(items)} remaining.")

    qid_counts = Counter()
    mention_rows = []       # one row per extracted mention (resolved or not), since the last flush
    entity_info = {}        # qid -> {qid, label, description}, deduped since the last flush

    def flush():
        if entity_info:
            for batch in _batched(list(entity_info.values()), BATCH_SIZE):
                client.run_write(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {qid: row.qid})
                    SET e.label = coalesce(row.label, e.label),
                        e.description = coalesce(row.description, e.description)
                    """,
                    rows=batch,
                )
        if mention_rows:
            for batch in _batched(mention_rows, BATCH_SIZE):
                client.run_write(
                    f"""
                    UNWIND $rows AS row
                    MATCH (src:{node_label} {{id: row.doc_id}})
                    CREATE (m:Mention {{text: row.mtext, lang: row.lang, mention_group: row.egroup, source: row.source}})
                    CREATE (src)-[:HAS_MENTION]->(m)
                    WITH src, m, row
                    FOREACH (_ IN CASE WHEN row.qid IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (e:Entity {{qid: row.qid}})
                        CREATE (m)-[:LINKS_TO]->(e)
                        MERGE (src)-[:MENTIONS]->(e)
                    )
                    """,
                    rows=batch,
                )
        entity_info.clear()
        mention_rows.clear()

    for i, item in enumerate(tqdm(todo, desc=f"Entity extraction+linking ({node_label})"), start=1):
        doc_id = item[id_field]
        text = item[text_field]
        lang = data_loader.detect_lang(text)

        tqdm.write(f"\n{'=' * 100}")
        tqdm.write(f"[{node_label}] {doc_id}")
        tqdm.write(f"  text: {text[:200]!r}{'...' if len(text) > 200 else ''}")
        tqdm.write(f"  detected language: {lang}")

        ner = entity_linker._get_ner()
        ner_raw = ner(text[:512]) if ner else []

        mention_meta = []
        seen = set()
        for r in ner_raw:
            start, end = entity_linker._expand_to_word_boundary(text[:512], r["start"], r["end"])
            mtext = text[:512][start:end].strip()
            tqdm.write(f"  ner span[{r['start']}:{r['end']}] -> expanded[{start}:{end}] = {mtext!r}  "
                       f"group={r.get('entity_group', 'MISC')}  score={float(r.get('score', 0)):.3f}")
            if len(mtext) <= 1 or mtext in seen:
                continue
            seen.add(mtext)
            mention_meta.append((mtext, r.get("entity_group", "MISC")))

        tqdm.write(f"  deduped mentions: {mention_meta}")

        if not mention_meta:
            tqdm.write("  no mentions extracted -- skipping entity linking for this item.")
            continue

        results = resolve_document(mention_meta, lang, full_text=text, enable_llm_fallback=True,
                                     enable_llm_tier2=enable_llm_tier2, pool_id_to_text=pool_id_to_text)

        for mtext, egroup in mention_meta:
            r = results[mtext]
            qid = r["qid"]
            mention_rows.append({
                "doc_id": doc_id, "mtext": mtext, "lang": lang,
                "egroup": egroup, "source": r["source"], "qid": qid,
            })
            if qid is None:
                continue  # unresolved mention -- Mention node still written, just no LINKS_TO/Entity
            qid_counts[qid] += 1
            if qid not in entity_info or not entity_info[qid].get("label"):
                entity_info[qid] = {"qid": qid, "label": r.get("label"), "description": r.get("description")}

        if i % LINKING_FLUSH_EVERY == 0:
            flush()

    flush()  # remainder -- fewer than LINKING_FLUSH_EVERY items since the last flush
    return qid_counts


def main():
    train, dev, pool = data_loader.load_all()
    if not pool:
        print("[kg_builder_st1] No pool data found -- place evidence_pool.json "
              "in the data/ directory first.")
        return

    client = Neo4jClient()
    client.setup_schema()

    print("\n=== Stage 1: Evidence nodes + embeddings ===")
    ingest_evidence_nodes(client, pool)

    if train or dev:
        print("\n=== Stage 2: Claim nodes + embeddings + gold edges ===")
        ingest_claim_nodes(client, train, dev)

    print("\n=== Stage 3: Entity extraction + linking (Evidence) ===")
    ingest_entities_and_mentions(
        client, pool, id_field="ID", text_field="Evidence", node_label="Evidence"
    )

    # BUG FIX 2026-07-10: this used to process only `train` claims, which left
    # every dev Claim node with zero MENTIONS edges. KGBoostedRetriever's
    # entity-overlap boost (and its cluster penalty, which fires whenever
    # overlap == 0) both depend on Claim MENTIONS edges existing -- so with
    # dev claims unlinked, the ablation/eval scripts were silently scoring a
    # KG-boost that could never see any entity overlap on the only split that
    # matters for reporting. Both train and dev claims need this step.
    all_claims = train + dev
    if all_claims:
        print("\n=== Stage 3b: Entity extraction + linking (Claims) ===")
        # Tier-2 LLM escalation (Claims only, see ingest_entities_and_mentions'
        # docstring) needs the pool embedded in memory for the top-3
        # related-evidence lookup -- prepare_pool_embeddings() is a no-op
        # if the disk cache already has them (it does, from Stage 1).
        from subtask1 import embeddings
        embeddings.prepare_pool_embeddings(pool)
        pool_id_to_text = {p["ID"]: p["Evidence"] for p in pool}
        ingest_entities_and_mentions(
            client, all_claims, id_field="ID", text_field="Text", node_label="Claim",
            enable_llm_tier2=True, pool_id_to_text=pool_id_to_text,
        )

    print("\n[kg_builder_st1] Done.")
    # OPTIONAL MATCH throughout -- a plain MATCH on any one node type (e.g.
    # Entity, which can legitimately be 0 if entity linking hit an outage)
    # collapses the whole chained query to zero rows, making this summary
    # silently print "empty" even when thousands of Evidence/Claim nodes
    # were written successfully in the same run.
    stats = client.run("""
        OPTIONAL MATCH (e:Evidence) WITH count(e) AS n_evidence
        OPTIONAL MATCH (c:Claim) WITH n_evidence, count(c) AS n_claim
        OPTIONAL MATCH (ent:Entity) RETURN n_evidence, n_claim, count(ent) AS n_entity
    """)
    print("Final KG stats:", dict(stats[0]) if stats else "empty")
    client.close()


if __name__ == "__main__":
    main()
