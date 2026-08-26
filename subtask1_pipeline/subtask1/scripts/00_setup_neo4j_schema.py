import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subtask1.neo4j_client import Neo4jClient

if __name__ == "__main__":
    client = Neo4jClient()
    client.setup_schema()
    result = client.run("MATCH (n) RETURN count(n) AS node_count")
    print(f"Neo4j is reachable. Current node count: {result[0]['node_count']}")
    client.close()
