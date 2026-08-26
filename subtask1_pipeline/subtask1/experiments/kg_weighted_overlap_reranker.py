OVERLAP_WEIGHT = 0.02


def _entity_document_frequency_overlap(client, claim_id: str, candidate_ids: list) -> dict:
    """For each candidate evidence sharing entities with the claim, sum
    1/log(doc_freq + e) over every shared entity -- doc_freq is how many
    Evidence+Claim documents mention that entity across the whole
    corpus. An entity that appears in few documents contributes close to
    1; one that appears in many contributes close to 0. Single Cypher
    query, same 3-hop traversal pattern as
    KGBoostedRetriever._entity_overlap_counts(), just weighted."""
    if not candidate_ids:
        return {}
    rows = client.run(
        """
        MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(ent:Entity)<-[:MENTIONS]-(ev:Evidence)
        WHERE ev.id IN $candidate_ids
        MATCH (ent)<-[:MENTIONS]-(doc)
        WITH ev, ent, count(DISTINCT doc) AS doc_freq
        RETURN ev.id AS evidence_id, sum(1.0 / log(doc_freq + 2.718281828)) AS weighted_overlap
        """,
        claim_id=claim_id, candidate_ids=candidate_ids,
    )
    return {r["evidence_id"]: r["weighted_overlap"] for r in rows}


def score_precomputed_weighted_overlap(client, claim_id: str, candidates: list, top_k: int = 3,
                                        weight: float = OVERLAP_WEIGHT) -> list:
    """Same input/output shape as KGBoostedRetriever.score_precomputed()
    (Dense score + entity-overlap boost) -- no cluster-penalty here,
    since that experiment already didn't hold up at production scale
    today. candidates: list of (evidence_id, dense_score) tuples.

    weight defaults to OVERLAP_WEIGHT (0.02, the measured best) --
    deliberately NOT config.ENTITY_OVERLAP_WEIGHT (0.2), which was tuned
    for flat-count overlap and is far too strong for this IDF-weighted
    signal (see the weight sweep in this module's docstring)."""
    if not candidates:
        return []
    candidate_ids = [c[0] for c in candidates]
    dense_scores = dict(candidates)

    weighted_overlap = _entity_document_frequency_overlap(client, claim_id, candidate_ids)

    final_scores = {}
    for eid in candidate_ids:
        score = dense_scores[eid]
        score += weight * weighted_overlap.get(eid, 0.0)
        final_scores[eid] = score

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in ranked[:top_k]]
