import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests

import config


def check_wikidata():
    print("Checking Wikidata SPARQL endpoint...")
    try:
        from SPARQLWrapper import JSON, SPARQLWrapper
        sparql = SPARQLWrapper(config.SPARQL_ENDPOINT, agent=config.SPARQL_USER_AGENT)
        sparql.setQuery('SELECT ?x WHERE {?x ?y ?z} LIMIT 1')
        sparql.setReturnFormat(JSON)
        t0 = time.time()
        sparql.query().convert()
        print(f"  OK ({time.time() - t0:.2f}s)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def check_huggingface():
    print("Checking HuggingFace Hub connectivity (tokenizer download)...")
    from transformers import AutoTokenizer
    all_ok = True
    # Check every model config.py actually points at, not just NLI_MODEL --
    # NER_MODEL and GEN_MODEL are gated repos on HF and fail independently
    # of the NLI model, so checking only one gives a false PASS.
    for label, model_name in [
        ("NLI_MODEL", config.NLI_MODEL),
        ("NER_MODEL", config.NER_MODEL),
        ("EMBED_MODEL", config.EMBED_MODEL),
        ("GEN_MODEL", config.GEN_MODEL),
    ]:
        try:
            t0 = time.time()
            AutoTokenizer.from_pretrained(model_name)
            print(f"  OK ({time.time() - t0:.2f}s) -- {label} = {model_name}")
        except Exception as e:
            print(f"  FAILED -- {label} = {model_name}: {e}")
            print(f"  If this is a 'gated repo' error, accept the license on the "
                  f"model page and set HF_TOKEN (huggingface-cli login) before retrying.")
            all_ok = False
    return all_ok


def check_wikipedia():
    print("Checking Wikipedia API...")
    try:
        t0 = time.time()
        r = requests.get("https://en.wikipedia.org/w/api.php",
                          params={"action": "query", "titles": "India", "format": "json"},
                          headers={"User-Agent": config.SPARQL_USER_AGENT},
                          timeout=10)
        r.raise_for_status()
        print(f"  OK ({time.time() - t0:.2f}s)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def check_neo4j():
    print("Checking Neo4j connection...")
    try:
        from subtask1.neo4j_client import Neo4jClient
        client = Neo4jClient()
        client.run("RETURN 1")
        client.close()
        print("  OK")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Did you run `docker-compose up -d` first?")
        return False


if __name__ == "__main__":
    results = {
        "Neo4j": check_neo4j(),
        "Wikidata SPARQL": check_wikidata(),
        "HuggingFace Hub": check_huggingface(),
        "Wikipedia API": check_wikipedia(),
    }
    print("\n=== Gate 1 summary ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(results.values()):
        print("\nFix the failed checks above before proceeding to script 02.")
        sys.exit(1)
    print("\nAll connectivity checks passed. Proceed to script 02.")
