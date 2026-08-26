from contextlib import contextmanager

from neo4j import GraphDatabase

import config


class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None, database=None):
        if not (password or config.NEO4J_PASSWORD):
            raise ValueError("Neo4j password is empty. Set NEO4J_PASSWORD in your .env file before connecting.")
        self.driver = GraphDatabase.driver(
            uri or config.NEO4J_URI,
            auth=(user or config.NEO4J_USER, password or config.NEO4J_PASSWORD),
        )
        # database=None means "use the server's default database" (the real
        # graph). Passing an explicit name (e.g. "testkg") targets a fully
        # separate database on the same server -- used for smoke-testing the
        # pipeline on a small sample without touching the real KG at all.
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
    # Schema setup -- run once via scripts/00_setup_neo4j_schema.py
    # -------------------------------------------------------------
    def setup_schema(self):
        statements = [
            "CREATE CONSTRAINT entity_qid IF NOT EXISTS FOR (n:Entity) REQUIRE n.qid IS UNIQUE",
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (n:EventCluster) REQUIRE n.cluster_id IS UNIQUE",
            "CREATE INDEX evidence_lang IF NOT EXISTS FOR (n:Evidence) ON (n.lang)",
            "CREATE INDEX claim_lang IF NOT EXISTS FOR (n:Claim) ON (n.lang)",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (n:Entity) ON (n.label)",
        ]
        for stmt in statements:
            self.run_write(stmt)
        print(f"[neo4j_client] Schema set up: {len(statements)} constraints/indexes.")

    # Optional scale-up path if the pool ever grows large enough that
    # brute-force cosine stops being instant. Uncomment and verify
    # syntax against your Neo4j version's docs before use.
    #
    # def setup_vector_index(self, label, prop, dims=config.EMBED_DIM):
    #     self.run_write(f"""
    #         CREATE VECTOR INDEX {label.lower()}_{prop}_idx IF NOT EXISTS
    #         FOR (n:{label}) ON (n.{prop})
    #         OPTIONS {{indexConfig: {{
    #             `vector.dimensions`: {dims},
    #             `vector.similarity_function`: 'cosine'
    #         }}}}
    #     """)


if __name__ == "__main__":
    client = Neo4jClient()
    client.setup_schema()
    result = client.run("MATCH (n) RETURN count(n) AS node_count")
    print("Current node count:", result[0]["node_count"])
    client.close()
