"""
Run this once, after `docker-compose up -d` and confirming Neo4j is
reachable, to create the constraints/indexes entity_graph_kg.py relies
on. Safe to re-run any time (every statement is IF NOT EXISTS).

Adapted from the Subtask 1 repository's
subtask1/scripts/00_setup_neo4j_schema.py -- identical logic, only the
import path changed to match this project's package structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neo4j_client import Neo4jClient

if __name__ == "__main__":
    client = Neo4jClient()
    client.setup_schema()
    result = client.run("MATCH (n) RETURN count(n) AS node_count")
    print(f"Neo4j is reachable. Current node count: {result[0]['node_count']}")
    client.close()