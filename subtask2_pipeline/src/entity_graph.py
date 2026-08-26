"""
STEP 7: Entity Graph Ranking -- persistent, shared Neo4j Knowledge Graph.

This graph is meant to GROW across your entire run -- hundreds of
claims all writing into the same database, never cleared between
claims. Because Evidence and Entity nodes are keyed by stable values
(URL, normalized entity text) rather than anything claim-specific, the
same real-world entity or the same real-world article is one shared
node no matter how many different claims happen to mention or retrieve
it. After a real run, this is explorable in the Neo4j Browser
(http://localhost:7474) with queries like:

    MATCH (c:Claim)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(ev:Evidence)
    RETURN c, e, ev

Node schema:
    Claim    {id, text, lang, created_at}
    Evidence {url, title, snippet, retrieved_at}
    Entity   {normalized_text, original_text}

Relationships:
    (Claim)-[:MENTIONS]->(Entity)
    (Evidence)-[:MENTIONS]->(Entity)
    (Evidence)-[:MATCHES {overlap_raw, overlap_norm, entity_rank}]->(Claim)

No SUPPORTS/REFUTES edges are ever written here -- this stage only
ranks candidate evidence; the actual verdict doesn't exist yet (that's
decided after the LLM reasoning stage, further down your pipeline).
MATCHES only records "this evidence was ranked against this claim by
entity overlap," nothing about whether it supports or refutes it.

created_at / retrieved_at are set ONLY on first creation (Cypher's
ON CREATE SET) -- reprocessing a claim or re-retrieving an
already-seen URL updates its text/title/snippet, but never rewrites
when that node first entered the graph. That's what makes these
timestamps meaningful in an accumulating, never-cleared graph.

Entities are SURFACE-FORM only, via entity_extraction.py exactly as it
already exists -- no changes to that file's extraction/normalization
logic.

OPTIONAL ADD-ON (2026-07): cross-language QID matching via
entity_linking.py. Surface-form matching alone can never equate two
different scripts (e.g. English "Bamangachi" vs. Bengali "বামনগাছি"),
so entity_linking.link_entity() is used, PURELY AS AN EXTRA SIGNAL, to
also try resolving claim/evidence entities to Wikidata QIDs and
counting QID-level overlap. This ADDS a parallel `entity_overlap_qid`
count alongside the existing `entity_overlap_raw` (surface-form) count
-- it does not replace or recompute the surface-form count, and if
Wikidata linking fails/times out/finds nothing (entity_linking.py NEVER
raises), the behavior is identical to before this add-on existed:
ranking falls back to `entity_overlap_raw` alone. `entity_rank` (the
only field fusion_ranking.py actually consumes) is derived from
whichever count is higher per item, so RRF's input contract is
unchanged and fusion_ranking.py itself is not touched at all.

Does NOT modify BM25 or embedding fields -- only appends
entity_overlap_raw / entity_overlap_norm / entity_overlap_qid /
entity_rank, exactly like bm25.py and embedding_rerank.py append their
own fields. entity_rank plus bm25_rank/embedding_rank are exactly what
the Reciprocal Rank Fusion stage needs (RRF uses ranks only, never raw
scores -- see module-level note in rank_by_entity_graph()).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import src.entity_extraction as entity_extraction
from src.neo4j_client import Neo4jClient

# Optional add-on -- see module docstring. Import failure (e.g. the
# `requests` dependency missing, though it's already required by
# article_fetcher.py) degrades this stage to pure surface-form matching,
# identical to pre-linking behavior, rather than crashing entity_graph.py.
try:
    from src import entity_linking
    _LINKING_AVAILABLE = True
except Exception as _e:
    print(f"[entity_graph] WARNING: entity_linking unavailable ({_e}); "
          f"falling back to surface-form-only entity matching.")
    entity_linking = None
    _LINKING_AVAILABLE = False


def _now_iso() -> str:
    """UTC timestamp as an ISO-8601 string -- stored as a plain string
    property (not Neo4j's native temporal type) so it round-trips
    through the Python driver and into JSON output with zero special
    handling on either side."""
    return datetime.now(timezone.utc).isoformat()


def _minmax_normalize(scores: list) -> list:
    """Scales raw overlap counts to 0-1. If all scores are equal
    (including all-zero -- the common case when NO entity overlaps at
    all), returns all 0.5 to avoid divide-by-zero / false signal --
    same convention used in bm25.py and embedding_rerank.py."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _write_claim_node(client: Neo4jClient, claim_id: str, claim_text: str, lang: str) -> None:
    client.run_write(
        """
        MERGE (c:Claim {id: $id})
        ON CREATE SET c.created_at = $created_at
        SET c.text = $text, c.lang = $lang
        """,
        id=claim_id, text=claim_text, lang=lang, created_at=_now_iso(),
    )


def _write_evidence_node(client: Neo4jClient, url: str, title: str, snippet: str) -> None:
    client.run_write(
        """
        MERGE (ev:Evidence {url: $url})
        ON CREATE SET ev.retrieved_at = $retrieved_at
        SET ev.title = $title, ev.snippet = $snippet
        """,
        url=url, title=title, snippet=snippet, retrieved_at=_now_iso(),
    )


def _write_claim_mentions(client: Neo4jClient, claim_id: str, entities: list,
                           lang: str = "english") -> list:
    """Returns the entity rows written (now includes optional `qid`),
    so the caller can reuse them for QID-overlap counting without a
    second extraction/linking pass."""
    rows = _entity_rows(entities, lang=lang)
    if not rows:
        return rows
    client.run_write(
        """
        UNWIND $rows AS row
        MERGE (e:Entity {normalized_text: row.normalized_text})
        ON CREATE SET e.original_text = row.original_text
        SET e.qid = coalesce(row.qid, e.qid)
        WITH e
        MATCH (c:Claim {id: $claim_id})
        MERGE (c)-[:MENTIONS]->(e)
        """,
        rows=rows, claim_id=claim_id,
    )
    return rows


def _write_evidence_mentions(client: Neo4jClient, url: str, entities: list,
                              lang: str = "english") -> list:
    """Returns the entity rows written (now includes optional `qid`),
    same reuse rationale as _write_claim_mentions above."""
    rows = _entity_rows(entities, lang=lang)
    if not rows:
        return rows
    client.run_write(
        """
        UNWIND $rows AS row
        MERGE (e:Entity {normalized_text: row.normalized_text})
        ON CREATE SET e.original_text = row.original_text
        SET e.qid = coalesce(row.qid, e.qid)
        WITH e
        MATCH (ev:Evidence {url: $url})
        MERGE (ev)-[:MENTIONS]->(e)
        """,
        rows=rows, url=url,
    )
    return rows


def _entity_rows(entities: list, lang: str = "english") -> list:
    """Converts raw extracted entity strings into
    {normalized_text, original_text, qid} rows, deduplicated by
    normalized form, dropping anything that normalizes to empty.

    `qid` is ADDITIVE and OPTIONAL (added 2026-07): None whenever
    entity_linking is unavailable or a lookup fails/finds nothing --
    existing callers/consumers that only ever read normalized_text /
    original_text are completely unaffected by this extra key."""
    rows, seen = [], set()
    for e in entities:
        key = entity_extraction.normalize_entity_key(e)
        if not key or key in seen:
            continue
        seen.add(key)
        qid = None
        if _LINKING_AVAILABLE:
            try:
                qid = entity_linking.link_entity(e, lang=lang)
            except Exception as ex:
                # Belt-and-braces -- entity_linking.link_entity() already
                # guarantees it never raises, but a failure here must
                # still never break graph writes.
                print(f"[entity_graph] WARNING: QID lookup failed for {e!r} ({ex}); "
                      f"continuing with surface-form matching only.")
                qid = None
        rows.append({"normalized_text": key, "original_text": e, "qid": qid})
    return rows


def _qid_overlap_count(claim_rows: list, evidence_rows: list) -> int:
    """Counts how many DISTINCT claim QIDs also appear among evidence
    QIDs. Rows with qid=None (no link found) never match here -- this
    is a purely ADDITIVE signal on top of the existing surface-form
    entity_overlap_count(); it never subtracts from or overrides it."""
    claim_qids = {r["qid"] for r in claim_rows if r.get("qid")}
    if not claim_qids:
        return 0
    evidence_qids = {r["qid"] for r in evidence_rows if r.get("qid")}
    return len(claim_qids & evidence_qids)


def _write_matches_edge(client: Neo4jClient, url: str, claim_id: str,
                         overlap_raw: int, overlap_norm: float, entity_rank: int) -> None:
    """Written for EVERY evidence item processed against this claim --
    not gated by a minimum overlap -- because entity_rank must exist
    for every item for Reciprocal Rank Fusion downstream to have a
    complete rank for every candidate, including ones with zero entity
    overlap (RRF still needs their rank, even if it's last)."""
    client.run_write(
        """
        MATCH (ev:Evidence {url: $url}), (c:Claim {id: $claim_id})
        MERGE (ev)-[m:MATCHES]->(c)
        SET m.overlap_raw = $overlap_raw, m.overlap_norm = $overlap_norm, m.entity_rank = $entity_rank
        """,
        url=url, claim_id=claim_id,
        overlap_raw=overlap_raw, overlap_norm=overlap_norm, entity_rank=entity_rank,
    )


def _query_overlap_counts(client: Neo4jClient, claim_id: str) -> dict:
    """The actual Cypher query computing entity overlap -- this is the
    real graph traversal your supervisor can see run in the Neo4j
    Browser, not a Python-side set intersection."""
    rows = client.run(
        """
        MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(ev:Evidence)
        RETURN ev.url AS url, count(DISTINCT e) AS overlap_count
        """,
        claim_id=claim_id,
    )
    return {r["url"]: r["overlap_count"] for r in rows}


def rank_by_entity_graph(
    client: Neo4jClient,
    claim_id: str,
    claim_text: str,
    lang: str,
    ranked_evidence: list,
) -> list:
    """Writes claim + evidence + entity nodes/edges into the shared,
    persistent graph (never cleared), queries real entity overlap via
    Cypher, records a MATCHES edge (with overlap_raw/overlap_norm/
    entity_rank) for every evidence item, and returns ranked_evidence
    with those same three fields appended in the JSON -- PLUS the new
    optional `entity_overlap_qid` field (see module docstring). Every
    existing field (title, url, snippet, bm25_*, embedding_*) is
    preserved unchanged.

    client: an already-connected Neo4jClient (dependency injected --
    callers control connection lifetime, so a batch run over hundreds
    of claims can share ONE connection instead of reconnecting per claim).
    """
    if not ranked_evidence:
        return []

    # 1. Claim node + its entities (rows now optionally carry a `qid`
    # from entity_linking.py -- see _entity_rows()).
    _write_claim_node(client, claim_id, claim_text, lang)
    claim_entities = entity_extraction.extract_entities(claim_text)
    claim_rows = _write_claim_mentions(client, claim_id, claim_entities, lang=lang)

    # 2. Evidence nodes + their entities, keyed by URL -- the same
    # stable identifier merge_dedup.py already dedupes on, so the same
    # real-world source is one shared node even across different claims.
    evidence_rows_by_url = {}
    for item in ranked_evidence:
        url = item.get("url", "")
        if not url:
            continue  # can't key a node without a stable id -- skip
        title, snippet = item.get("title", ""), item.get("snippet", "")
        _write_evidence_node(client, url, title, snippet)
        evidence_entities = entity_extraction.extract_entities(f"{title} {snippet}")
        evidence_rows_by_url[url] = _write_evidence_mentions(
            client, url, evidence_entities, lang=lang
        )

    # 3. Real Cypher query for surface-form overlap -- UNCHANGED, not a
    # Python set intersection. This remains the primary signal.
    overlap_by_url = _query_overlap_counts(client, claim_id)

    # 3b. OPTIONAL ADD-ON: QID overlap per evidence item, computed in
    # Python from the rows already written in step 1/2 above (no extra
    # graph traversal needed -- QIDs were already resolved during
    # linking, this just counts set intersections). If linking was
    # unavailable/failed for everything, every qid is None and this
    # naturally contributes 0 everywhere -- i.e. behaves exactly like
    # before this add-on existed.
    qid_overlap_by_url = {
        url: _qid_overlap_count(claim_rows, rows)
        for url, rows in evidence_rows_by_url.items()
    }

    # 4. Combine: an evidence item's EFFECTIVE overlap for ranking
    # purposes is the max of its surface-form count and its QID count --
    # this rewards genuine cross-language matches (which surface-form
    # alone would score as 0) without ever scoring LOWER than the
    # pre-existing surface-form-only behavior. Normalize + rank in
    # Python, same convention as bm25.py / embedding_rerank.py
    # (min-max, all-equal -> 0.5).
    surface_scores = [overlap_by_url.get(item.get("url", ""), 0) for item in ranked_evidence]
    qid_scores = [qid_overlap_by_url.get(item.get("url", ""), 0) for item in ranked_evidence]
    effective_scores = [max(s, q) for s, q in zip(surface_scores, qid_scores)]
    norm_scores = _minmax_normalize(effective_scores)

    combined = sorted(
        zip(ranked_evidence, surface_scores, qid_scores, effective_scores, norm_scores),
        key=lambda x: -x[3],
    )

    ranked = []
    for rank, (item, surface_raw, qid_raw, effective_raw, norm) in enumerate(combined, start=1):
        new_item = dict(item)  # copy -- preserves every existing field untouched
        new_item["entity_overlap_raw"] = surface_raw       # UNCHANGED semantics
        new_item["entity_overlap_qid"] = qid_raw            # NEW, additive, informational
        new_item["entity_overlap_norm"] = norm              # now normalized from the combined signal
        new_item["entity_rank"] = rank                      # the only field fusion_ranking.py reads
        ranked.append(new_item)

        # 5. Persist this ranking back into the graph as a MATCHES edge.
        # NOT a SUPPORTS/REFUTES edge -- the verdict isn't known yet.
        # Stored overlap_raw is the effective (combined) score, matching
        # what actually drove entity_rank, for a graph that's queryable
        # and consistent with the JSON output.
        url = item.get("url", "")
        if url:
            _write_matches_edge(client, url, claim_id, effective_raw, norm, rank)

    return ranked


def rank_claim_folder(
    claim_folder: str,
    client: Neo4jClient = None,
    retrieval_filename: str = "retrieval_output.json",
    embedding_filename: str = "embedding_ranked.json",
    output_filename: str = "entity_ranked.json",
) -> dict:
    """Convenience wrapper for running this stage on ONE claim's output
    folder. claim_folder is supplied by the caller -- nothing here is
    hardcoded to a specific claim.

    If client is not supplied, opens and closes its own connection for
    this one claim (fine for a single test run). When processing many
    claims, pass in ONE shared client instead (see
    scripts/build_entity_graph_batch.py) -- the whole point of this
    stage is a graph that accumulates across claims in one shared
    database, not a fresh connection (and implicitly, fine to keep
    reconnecting, but wasteful) per claim.

    Reads:  <claim_folder>/retrieval_output.json  (for claim id/text/lang)
            <claim_folder>/embedding_ranked.json   (for reranked_evidence)
    Writes: <claim_folder>/entity_ranked.json
    """
    folder = Path(claim_folder)

    with open(folder / retrieval_filename, encoding="utf-8") as f:
        retrieval_data = json.load(f)
    with open(folder / embedding_filename, encoding="utf-8") as f:
        embedding_data = json.load(f)

    claim_meta = embedding_data["claim"]
    claim_id, claim_text, lang = claim_meta["id"], claim_meta["text"], claim_meta["lang"]
    reranked_evidence = embedding_data["reranked_evidence"]

    owns_client = client is None
    if owns_client:
        client = Neo4jClient()
        client.setup_schema()

    try:
        entity_ranked = rank_by_entity_graph(client, claim_id, claim_text, lang, reranked_evidence)
    finally:
        if owns_client:
            client.close()

    output = {"claim": claim_meta, "entity_ranked_evidence": entity_ranked}

    output_path = folder / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[entity_graph_kg] Saved {len(entity_ranked)} entity-ranked item(s) -> {output_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Neo4j-backed entity graph ranking on one claim's output folder. "
                    "No claim ID or file path is hardcoded -- pass the folder to process. "
                    "Requires Neo4j running (see docker-compose.yml) and NEO4J_PASSWORD set."
    )
    parser.add_argument(
        "claim_folder",
        help="Path to a single claim's output folder, e.g. outputs/claims/S2_T_BN_1001 "
             "(must already contain retrieval_output.json and embedding_ranked.json).",
    )
    args = parser.parse_args()

    rank_claim_folder(args.claim_folder)