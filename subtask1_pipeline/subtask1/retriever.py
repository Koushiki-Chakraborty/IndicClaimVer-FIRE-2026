import numpy as np
from rank_bm25 import BM25Okapi

import config
from subtask1 import data_loader
from subtask1.neo4j_client import Neo4jClient
from subtask1 import embeddings


# ---------------------------------------------------------------------
# 1. BM25 baseline
# ---------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, pool: list):
        self.pool_by_lang = data_loader.build_pool_by_lang(pool)
        self.bm25_by_lang = {}
        for lang, items in self.pool_by_lang.items():
            # BEHAVIOR NOTE: .lower() added 2026-07-06 to match the validated
            # Gate 2 script's tokenize() = text.lower().split(). Previously
            # this was item["Evidence"].split() (case-sensitive), which did NOT
            # match the already-confirmed 0.768/0.563 numbers. This fix is
            # required for BM25Retriever to reproduce the Gate 2 baseline.
            corpus = [item["Evidence"].lower().split() for item in items]
            self.bm25_by_lang[lang] = BM25Okapi(corpus)

    def retrieve(self, claim_text: str, top_k: int = config.TOP_K_RETRIEVE) -> list:
        lang = data_loader.detect_lang(claim_text)
        if lang not in self.bm25_by_lang:
            return []
        items = self.pool_by_lang[lang]
        # BEHAVIOR NOTE: .lower() added 2026-07-06 to match Gate 2 tokenize().
        # Previously claim_text.split() (case-sensitive).
        scores = self.bm25_by_lang[lang].get_scores(claim_text.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [items[i]["ID"] for i in ranked[:top_k]]


# ---------------------------------------------------------------------
# 2. Dense baseline (embeddings pulled from Neo4j)
# ---------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, client: Neo4jClient):
        self.client = client
        self._cache = {}  # lang -> (ids, matrix)

    def _get_lang_matrix(self, lang: str):
        if lang in self._cache:
            return self._cache[lang]
        rows = self.client.run(
            "MATCH (e:Evidence {lang: $lang}) RETURN e.id AS id, e.embedding AS embedding",
            lang=lang,
        )
        ids = [r["id"] for r in rows]
        matrix = np.array([r["embedding"] for r in rows]) if rows else np.zeros((0, config.EMBED_DIM))
        self._cache[lang] = (ids, matrix)
        return ids, matrix

    def retrieve(self, claim_text: str, top_k: int = config.TOP_K_RETRIEVE,
                 return_scores: bool = False):
        lang = data_loader.detect_lang(claim_text)
        ids, matrix = self._get_lang_matrix(lang)
        if len(ids) == 0:
            return [] if not return_scores else []
        query_vec = embeddings.encode(claim_text)
        sims = embeddings.cosine_sim_matrix(query_vec, matrix)
        ranked_idx = np.argsort(-sims)[:top_k]
        if return_scores:
            return [(ids[i], float(sims[i])) for i in ranked_idx]
        return [ids[i] for i in ranked_idx]


# ---------------------------------------------------------------------
# 3. KG-boosted retrieval
# ---------------------------------------------------------------------
class KGBoostedRetriever:
    def __init__(self, client: Neo4jClient, dense: DenseRetriever):
        self.client = client
        self.dense = dense

    def _entity_overlap_counts(self, claim_id: str, candidate_ids: list) -> dict:
        """One Cypher query: for a given Claim, how much entity overlap
        does it share with each candidate Evidence -- weighted by how
        often each shared entity appears across the corpus (inverse
        document frequency), not a flat per-entity count. A flat count
        let common entities (Delhi, Modi -- appear in thousands of
        documents) dominate the score the same as a specific,
        infrequent entity would, even though the infrequent one is much
        stronger evidence of genuine relevance -- measured directly:
        flat-count KG-boost underperformed plain Dense (Success@3
        0.7834 vs 0.8312 on all 5,670 train+dev claims). Weighting
        overlap by inverse document frequency fixed that (Success@3
        0.8395, beating Dense -- see subtask1/experiments/README.md for
        the full weight sweep that found ENTITY_OVERLAP_WEIGHT=0.02 as
        the best point)."""
        if not candidate_ids:
            return {}
        rows = self.client.run(
            """
            MATCH (c:Claim {id: $claim_id})-[:MENTIONS]->(ent:Entity)<-[:MENTIONS]-(ev:Evidence)
            WHERE ev.id IN $candidate_ids
            MATCH (ent)<-[:MENTIONS]-(doc)
            WITH ev, ent, count(DISTINCT doc) AS doc_freq
            RETURN ev.id AS evidence_id, sum(1.0 / log(doc_freq + 2.718281828)) AS overlap
            """,
            claim_id=claim_id, candidate_ids=candidate_ids,
        )
        return {r["evidence_id"]: r["overlap"] for r in rows}

    def _cluster_membership(self, candidate_ids: list) -> dict:
        if not candidate_ids:
            return {}
        rows = self.client.run(
            """
            MATCH (ev:Evidence)-[:IN_CLUSTER]->(cl:EventCluster)
            WHERE ev.id IN $candidate_ids
            RETURN ev.id AS evidence_id, cl.cluster_id AS cluster_id, cl.density_score AS density
            """,
            candidate_ids=candidate_ids,
        )
        return {r["evidence_id"]: (r["cluster_id"], r["density"]) for r in rows}

    def score_precomputed(self, claim_id: str, candidates: list,
                           top_k: int = config.TOP_K_FINAL) -> list:
        """Same overlap scoring as retrieve(), but against a Dense
        candidate list that was already computed elsewhere (e.g. a bulk
        dense-retrieve-all pass), instead of calling self.dense.retrieve()
        again. For pipelines that split "get everyone's Dense top-10" and
        "entity-link + KG-score" into separate phases/processes so the
        expensive linking step only ever touches the reduced union that
        showed up in some claim's top-10.

        candidates: list of (evidence_id, dense_score) tuples -- a claim's
        Dense top-10, already computed.

        No cluster-penalty here -- tried it (see Known limitations),
        found it degraded results at production scale regardless of
        DBSCAN threshold tried (clustering kept collapsing into one giant
        cluster per language instead of isolating genuinely confusable
        sub-events), so it's not part of the validated, currently-active
        scoring. _cluster_membership() is kept in this class, unused, for
        future retuning rather than deleted."""
        if not candidates:
            return []
        candidate_ids = [c[0] for c in candidates]
        dense_scores = dict(candidates)

        overlap = self._entity_overlap_counts(claim_id, candidate_ids)

        final_scores = {}
        for eid in candidate_ids:
            score = dense_scores[eid]
            score += config.ENTITY_OVERLAP_WEIGHT * overlap.get(eid, 0.0)
            final_scores[eid] = score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in ranked[:top_k]]

    def retrieve(self, claim_id: str, claim_text: str,
                 top_k: int = config.TOP_K_FINAL) -> list:
        dense_candidates = self.dense.retrieve(
            claim_text, top_k=config.TOP_K_RETRIEVE, return_scores=True
        )
        return self.score_precomputed(claim_id, dense_candidates, top_k=top_k)


if __name__ == "__main__":
    train, dev, pool = data_loader.load_all()
    if pool:
        bm25 = BM25Retriever(pool)
        test_claim = dev[0]["Text"] if dev else train[0]["Text"]
        print("Test claim:", test_claim[:80])
        print("BM25 top-3:", bm25.retrieve(test_claim, top_k=3))
