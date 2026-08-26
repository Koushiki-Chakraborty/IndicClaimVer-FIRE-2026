import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1 import data_loader
from subtask1.neo4j_client import Neo4jClient
from subtask1.retriever import DenseRetriever


def main():
    print("=" * 60)
    print("Task 2 Probe: Neo4j Evidence node population check")
    print("=" * 60)

    client = Neo4jClient()

    # ------------------------------------------------------------------
    # Step 1: Count Evidence nodes in Neo4j
    # ------------------------------------------------------------------
    rows = client.run("MATCH (e:Evidence) RETURN count(e) AS n")
    n_evidence = rows[0]["n"] if rows else 0
    print(f"\n[Neo4j] Total Evidence nodes: {n_evidence}")

    # ------------------------------------------------------------------
    # Step 2: Check how many have an embedding set
    # ------------------------------------------------------------------
    rows_emb = client.run(
        "MATCH (e:Evidence) WHERE e.embedding IS NOT NULL RETURN count(e) AS n"
    )
    n_with_emb = rows_emb[0]["n"] if rows_emb else 0
    print(f"[Neo4j] Evidence nodes with embedding: {n_with_emb}")

    # ------------------------------------------------------------------
    # Step 3: Sample per-language counts
    # ------------------------------------------------------------------
    lang_rows = client.run(
        "MATCH (e:Evidence) RETURN e.lang AS lang, count(e) AS n ORDER BY n DESC"
    )
    print(f"[Neo4j] Per-language Evidence node counts:")
    if lang_rows:
        for r in lang_rows:
            print(f"  lang={r['lang']}  n={r['n']}")
    else:
        print("  (no rows -- Evidence nodes do not exist yet)")

    # ------------------------------------------------------------------
    # Step 4: Run DenseRetriever against 5 dev claims
    # ------------------------------------------------------------------
    print("\n--- DenseRetriever test: 5 dev claims ---")
    _, dev, _ = data_loader.load_all()
    if not dev:
        print("[ERROR] dev_subtask1.json not found -- cannot test DenseRetriever")
        client.close()
        return

    dense = DenseRetriever(client)
    for claim in dev[:5]:
        result = dense.retrieve(claim["Text"], top_k=3)
        print(f"  Claim ID={claim['ID']} lang={data_loader.detect_lang(claim['Text'])}")
        print(f"  Text (first 80 chars): {claim['Text'][:80]}")
        print(f"  DenseRetriever top-3: {result}")
        print()

    client.close()
    print("=" * 60)
    print("Probe complete. Decision needed: Neo4j-backed vs NumPy-backed dense retrieval.")
    print("If all DenseRetriever results above are [], Neo4j is NOT populated yet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
