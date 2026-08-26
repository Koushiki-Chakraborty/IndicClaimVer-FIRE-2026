"""
Neo4j connection wrapper + schema setup.

Adapted from the Subtask 1 repository's subtask1/neo4j_client.py.
Two changes from the original, both required, nothing else touched:

1. Import style: `import config` (Subtask 1's top-level style) ->
   `from . import config` (this project's package-relative style).

2. Schema constraints: Subtask 1 keyed Entity nodes on `qid` (a
   Wikidata identifier), because its entities were linked to Wikidata.
   This project's entities are SURFACE-FORM only (see entity_extraction.py
   -- no Wikidata linking, deliberately, to avoid SPARQL/network
   dependency for a stage that only needs overlap counting). So the
   Entity constraint here keys on `normalized_text` (a normalized
   surface string) instead of `qid`, and Evidence keys on `url`
   directly (no separate `id` property). The `EventCluster` constraint
   is dropped entirely -- event clustering isn't part of this pipeline.

Design decision carried over unmodified from Subtask 1: no Neo4j
vector index is set up. Entity overlap here is a simple MENTIONS-edge
count via Cypher, not a similarity search -- there's no embedding
stored on graph nodes to index in the first place.
"""
from contextlib import contextmanager

from neo4j import GraphDatabase
import src.config as config


class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None, database=None):
        if not (password or config.NEO4J_PASSWORD):
            raise ValueError("Neo4j password is empty. Set NEO4J_PASSWORD in your .env file before connecting.")
        self.driver = GraphDatabase.driver(
            uri or config.NEO4J_URI,
            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD),
        )
        # database=None means "use the server's default database".
        # Passing an explicit name (e.g. "testkg") targets a separate
        # database on the same server -- useful for smoke-testing
        # without touching your real graph.
        self.database = database or config.NEO4J_DATABASE or None

    def close(self):
        self.driver.close()

    @contextmanager
    def session(self):
        s = self.driver.session(database=self.database)
        try:
            yield s
        finally:
            s.close()

    def run(self, query: str, **params):
        with self.session() as s:
            return list(s.run(query, **params))

    def run_write(self, query: str, **params):
        with self.session() as s:
            return s.execute_write(lambda tx: list(tx.run(query, **params)))

    # -------------------------------------------------------------
    # Schema setup -- safe to call every run (all IF NOT EXISTS)
    # -------------------------------------------------------------
    def setup_schema(self):
        statements = [
            "CREATE CONSTRAINT entity_normalized_text IF NOT EXISTS FOR (n:Entity) REQUIRE n.normalized_text IS UNIQUE",
            "CREATE CONSTRAINT evidence_url IF NOT EXISTS FOR (n:Evidence) REQUIRE n.url IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX evidence_lang IF NOT EXISTS FOR (n:Evidence) ON (n.lang)",
            "CREATE INDEX claim_lang IF NOT EXISTS FOR (n:Claim) ON (n.lang)",
            "CREATE INDEX entity_original_text IF NOT EXISTS FOR (n:Entity) ON (n.original_text)",
        ]
        for stmt in statements:
            self.run_write(stmt)
        print(f"[neo4j_client] Schema set up: {len(statements)} constraints/indexes.")


if __name__ == "__main__":
    client = Neo4jClient()
    client.setup_schema()
    result = client.run("MATCH (n) RETURN count(n) AS node_count")
    print("Current node count:", result[0]["node_count"])
    client.close()