from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import config
from subtask1.neo4j_client import Neo4jClient

# eps is a cosine DISTANCE (1 - similarity), so a similarity threshold
# of 0.72 becomes eps=0.28: points closer than that count as neighbors.
DBSCAN_EPS = 1.0 - config.CLUSTER_SIMILARITY_THRESHOLD
DBSCAN_MIN_SAMPLES = config.CLUSTER_MIN_SIZE


def cluster_language_subset(evidence_ids: list, embeddings_matrix: np.ndarray) -> dict:
    """Returns {evidence_id: cluster_label} where label -1 means
    'not part of any dense cluster' (the majority case, expected)."""
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embeddings_matrix)
    return dict(zip(evidence_ids, labels))


def compute_density_score(cluster_vectors: np.ndarray) -> float:
    """Mean cosine similarity of each member to the cluster centroid.
    Higher = tighter, more confusable cluster = more disambiguation
    value if we can add it."""
    centroid = cluster_vectors.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    normed = cluster_vectors / (np.linalg.norm(cluster_vectors, axis=1, keepdims=True) + 1e-8)
    sims = normed @ centroid
    return float(sims.mean())


def build_event_clusters(client: Neo4jClient):
    print("[event_clustering] Fetching Evidence embeddings from Neo4j...")
    rows = client.run("MATCH (e:Evidence) RETURN e.id AS id, e.lang AS lang, e.embedding AS embedding")
    by_lang = defaultdict(lambda: {"ids": [], "vecs": []})
    for r in rows:
        by_lang[r["lang"]]["ids"].append(r["id"])
        by_lang[r["lang"]]["vecs"].append(r["embedding"])

    cluster_node_rows = []
    membership_rows = []
    cluster_counter = 0

    for lang, data in by_lang.items():
        ids = data["ids"]
        vecs = np.array(data["vecs"])
        print(f"[event_clustering] Clustering {len(ids)} '{lang}' evidence items "
              f"(eps={DBSCAN_EPS:.2f}, min_samples={DBSCAN_MIN_SAMPLES})...")
        labels = cluster_language_subset(ids, vecs)

        cluster_members = defaultdict(list)
        for eid, label in labels.items():
            if label != -1:
                cluster_members[label].append(eid)

        print(f"[event_clustering]   Found {len(cluster_members)} dense clusters "
              f"in '{lang}' (covering {sum(len(v) for v in cluster_members.values())}"
              f"/{len(ids)} items; the rest are unclustered -- expected and fine).")

        id_to_idx = {eid: i for i, eid in enumerate(ids)}
        for local_label, members in cluster_members.items():
            cluster_id = f"{lang}_cluster_{cluster_counter}"
            cluster_counter += 1
            member_vecs = vecs[[id_to_idx[m] for m in members]]
            density = compute_density_score(member_vecs)
            cluster_node_rows.append({
                "cluster_id": cluster_id,
                "size": len(members),
                "density_score": density,
                "lang": lang,
            })
            for m in members:
                membership_rows.append({"evidence_id": m, "cluster_id": cluster_id})

    print(f"[event_clustering] Writing {len(cluster_node_rows)} EventCluster nodes "
          f"and {len(membership_rows)} IN_CLUSTER edges...")
    for batch_start in tqdm(range(0, len(cluster_node_rows), 200), desc="Cluster nodes"):
        batch = cluster_node_rows[batch_start:batch_start + 200]
        client.run_write(
            """
            UNWIND $rows AS row
            MERGE (cl:EventCluster {cluster_id: row.cluster_id})
            SET cl.size = row.size, cl.density_score = row.density_score, cl.lang = row.lang
            """,
            rows=batch,
        )
    for batch_start in tqdm(range(0, len(membership_rows), 200), desc="Membership edges"):
        batch = membership_rows[batch_start:batch_start + 200]
        client.run_write(
            """
            UNWIND $rows AS row
            MATCH (e:Evidence {id: row.evidence_id})
            MATCH (cl:EventCluster {cluster_id: row.cluster_id})
            MERGE (e)-[:IN_CLUSTER]->(cl)
            """,
            rows=batch,
        )
    print("[event_clustering] Done.")


if __name__ == "__main__":
    client = Neo4jClient()
    build_event_clusters(client)
    client.close()
