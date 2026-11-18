import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Windows' console defaults to cp1252, which cannot encode Devanagari/
# Bengali characters -- and this pipeline prints Hindi/Bengali text
# constantly (entity mentions, claim/evidence snippets, warnings).
# Without this, any such print() crashes the whole process with
# UnicodeEncodeError the first time a non-ASCII string reaches stdout.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT / "cache"))
CACHE_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_PATH = DATA_DIR / "train_subtask1.json"
DEV_PATH = DATA_DIR / "dev_subtask1.json"
POOL_PATH = DATA_DIR / "evidence_pool.json"
SPARQL_CACHE_PATH = CACHE_DIR / "sparql_cache.json"

# ---------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
# None = server default database (the real graph). Set to e.g. "testkg" to
# target a separate database on the same server for smoke-testing.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or None

# ---------------------------------------------------------------------
# Models -- verify these actually load on your machine (Gate 1) before
# running anything downstream. This sandbox could not reach
# huggingface.co or query.wikidata.org to pre-validate these names.
# ---------------------------------------------------------------------
NER_MODEL = os.getenv("NER_MODEL", "ai4bharat/IndicNER")
# 2026-07-06: switched from paraphrase-multilingual-mpnet-base-v2 -- that
# model's own trained-language list (see sbert.net/docs/pretrained_models.html)
# does NOT include Bengali, which is exactly why Gate 2 showed catastrophic
# Bengali dense-retrieval Success@3 (0.207 vs BM25's 0.670). E5 models are
# trained on mC4, which does cover Bengali, and are tuned for asymmetric
# query/passage retrieval (see EMBED_NEEDS_E5_PREFIX below) -- a better fit
# for claim-vs-evidence search than mpnet's paraphrase-similarity objective.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_DIM = 768
# E5 models require "query: " / "passage: " text prefixes to perform as
# documented -- without them they still run, just noticeably worse. Only
# intfloat/e5-* and intfloat/multilingual-e5-* models need this.
EMBED_NEEDS_E5_PREFIX = EMBED_MODEL.startswith("intfloat/") and "e5" in EMBED_MODEL
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
GEN_MODEL = os.getenv("GEN_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

# Standalone binary SUPPORTS/REFUTES classifier -- fine-tuned MuRIL,
# trained in notebook/FIRE2026_FINAL.ipynb using the hyperparameters
# found by notebook/FIRE_2026_HPT_01.ipynb's Optuna search. There is no
# training .py module for this in the repo -- the checkpoint was
# uploaded to the Hub so subtask1/muril_classifier.py (used by
# scripts/07_eval_full_pipeline_st1.py) can load it directly instead of
# needing the original Colab output on disk.
MURIL_CLASSIFIER_MODEL = os.getenv("MURIL_CLASSIFIER_MODEL", "nirnit-13/muril-supports-refutes")

# LLM fallback for entity linking -- last-resort stage only, after the
# entire free Wikidata cascade (exact/code-mixed/morphology/wbsearch) has
# already failed. Paid API -- kept to a single call per weak mention,
# capped output tokens, disk-cached so a mention is never re-billed twice.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding caches are namespaced by model name so switching EMBED_MODEL
# (as just happened) can never silently mix stale vectors from a previous
# model into a cosine-similarity computation.
_embed_model_slug = EMBED_MODEL.replace("/", "__")
EMBED_CACHE_PATH = CACHE_DIR / f"embed_cache__{_embed_model_slug}.pkl"
POOL_EMBED_CACHE_PATH = CACHE_DIR / f"pool_embeddings__{_embed_model_slug}.pkl"

# ---------------------------------------------------------------------
# SPARQL politeness (Wikidata public endpoint has no hard published
# rate limit but will start returning 429s if hammered; be conservative)
# ---------------------------------------------------------------------
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_USER_AGENT = os.getenv(
    "SPARQL_USER_AGENT",
    "IndicClaimVer-Research/1.0 (research use; contact: your_email@example.com)",
)
SPARQL_SLEEP_SECONDS = float(os.getenv("SPARQL_SLEEP_SECONDS", "1.0"))
# A hung/slow response with no timeout can block a single query (and the
# whole entity-linking loop behind it) forever -- observed directly: a real
# smoketest run froze for 1h40m+ on one query. Bounded so a hang fails fast
# into the same try/except every caller already has, instead of hanging.
SPARQL_TIMEOUT_SECONDS = float(os.getenv("SPARQL_TIMEOUT_SECONDS", "30"))

# ---------------------------------------------------------------------
# COMPUTED from actual train_subtask1.json label counts
# (SUPPORTS=3040, REFUTES=1496, total=4536)
# weight_class = total / (num_classes * count_class)
# ---------------------------------------------------------------------
CLASS_WEIGHTS = {
    "SUPPORTS": 4536 / (2 * 3040),   # 0.746
    "REFUTES": 4536 / (2 * 1496),    # 1.516
}
LABEL2ID = {"REFUTES": 0, "SUPPORTS": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------------------------------------------------
# TUNABLE retrieval-boost weights.
# ---------------------------------------------------------------------
# VALIDATED: weight-swept on all 5,670 train+dev claims after switching
# _entity_overlap_counts() to overlap weighted by inverse document
# frequency (see retriever.py and the root README.md, Notes -> "KG weight sweep", for
# the full sweep). 0.02 was the
# best of every value tried in [0.0, 2.0], and every value in [0.01,
# 0.05] beat plain Dense retrieval on both Success@3 and nDCG@3 -- a
# real, robust region, not a single lucky point. 0.2 (the old default)
# was tuned for a flat per-entity count, not this weighted signal, and
# is far too strong for it.
ENTITY_OVERLAP_WEIGHT = 0.02
# Not currently applied (score_precomputed() no longer calls the
# cluster-penalty at all -- see Known limitations): tried retuning the
# DBSCAN threshold, it degraded results regardless, kept here only in
# case cluster-penalty scoring is revisited later.
EVENT_CLUSTER_PENALTY_WEIGHT = 0.15

# Event-density clustering thresholds (Gate 4). These decide what counts
# as a "confusable" cluster worth extra disambiguation. 0.72 was the
# starting point from the RG Kar analysis (278 evidence items, all one
# story -- a small, homogeneous set). At full 13,333-item production
# scale, across many different stories, that threshold was too loose --
# measured directly: it merged ALL evidence per language into one giant
# cluster each (100% coverage), which makes the cluster-penalty an
# undiscriminating blanket rather than a signal that flags specific
# confusable sub-events. Tightened to require much closer similarity
# before DBSCAN treats items as neighbors, to stop that chaining.
CLUSTER_MIN_SIZE = 15
CLUSTER_SIMILARITY_THRESHOLD = 0.85  # cosine distance threshold for clustering

# Retrieval
TOP_K_RETRIEVE = 10   # candidates considered before final top-3 cut
TOP_K_FINAL = 3        # what actually gets scored by Success@3 / nDCG@3
