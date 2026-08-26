"""
Runs entity_graph_kg's ranking stage across every claim folder that
already has embedding_ranked.json, writing all of them into the SAME
persistent Neo4j graph over ONE shared connection.

This is the script that actually produces the "hundreds of claims,
one accumulating graph" result -- running entity_graph_kg.py's CLI
per claim would technically also accumulate into the same graph (the
graph is never cleared regardless), but would open and close a new
Neo4j connection for every single claim, which is wasteful at scale.

Usage:
    python -m scripts.build_entity_graph_batch
    python -m scripts.build_entity_graph_batch --limit 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.entity_graph import rank_claim_folder
from src.neo4j_client import Neo4jClient


def find_ready_claim_folders(claims_dir: Path) -> list:
    """A claim folder is ready for this stage once it has both
    retrieval_output.json and embedding_ranked.json (produced by
    earlier pipeline stages)."""
    ready = []
    for folder in sorted(claims_dir.iterdir()):
        if not folder.is_dir():
            continue
        if (folder / "retrieval_output.json").exists() and (folder / "embedding_ranked.json").exists():
            ready.append(folder)
    return ready


def main():
    parser = argparse.ArgumentParser(
        description="Build the persistent entity graph across many claims, one shared Neo4j connection."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many claim folders. Omit to process all ready folders.")
    args = parser.parse_args()

    claim_folders = find_ready_claim_folders(config.CLAIMS_OUTPUT_DIR)
    if args.limit is not None:
        claim_folders = claim_folders[:args.limit]

    print(f"[build_entity_graph_batch] Found {len(claim_folders)} claim folder(s) ready for entity graph ranking.")

    client = Neo4jClient()
    client.setup_schema()

    succeeded, failed = 0, []
    try:
        for folder in claim_folders:
            try:
                rank_claim_folder(str(folder), client=client)
                succeeded += 1
            except Exception as e:
                print(f"[build_entity_graph_batch] ERROR on {folder.name}: {e}")
                failed.append(folder.name)
    finally:
        client.close()

    print(f"\n[build_entity_graph_batch] Done. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed claim folders: {failed}")


if __name__ == "__main__":
    main()