"""
Central configuration for the Subtask 2 pipeline (pre-KG stages only).
Everything that might need tuning lives here, not scattered across files.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env into the environment (os.environ) BEFORE
# anything below reads them. Without this call, python-dotenv being
# installed does nothing on its own -- .env is never read automatically.
load_dotenv()

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
# BUG FIX: this was "outputs" (plural) -- but every real processed claim
# (1009 test + 631 dev), the Wikidata QID cache, and both GPT_output/
# and classifier_output/ actually live on disk under "output" (singular).
# Every path in this file, and everywhere else in the codebase, derives
# from this one constant (verified: grep for OUTPUT_DIR / hardcoded
# "outputs" across every .py file -- nothing else hardcodes the string),
# so the old value meant a fresh `python run_pipeline.py` run would have
# created a brand-new, empty "outputs/" directory next to the real one
# and silently reprocessed all 1009 claims from scratch -- --skip-existing
# would never see the already-completed work, since it was never looking
# in the right place.
OUTPUT_DIR = ROOT_DIR / "output"

# Your 4 test files should live in DATA_DIR with these exact names:
DATA_FILES = {
    "english": DATA_DIR / "subtask2_english_test.json",
    "hindi": DATA_DIR / "subtask2_hindi_test.json",
    "bengali": DATA_DIR / "subtask2_bengali_test.json",
    "codemix": DATA_DIR / "subtask2_codemix_test.json",
}

# ----------------------------------------------------------------------
# API Keys (loaded from environment / .env — never hardcode keys here)
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
# CHANGED (Gemini -> GPT-4o mini swap, verification stage only): new key
# for src/verification.py's LLM client. GEMINI_API_KEY is left in place
# above -- unused now, but harmless, and avoids touching anything that
# might still reference it.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ----------------------------------------------------------------------
# LLM settings (used for query generation / translation)
# ----------------------------------------------------------------------
# CHANGED (2026-07): "gemini-2.5-flash" started returning 404 ("no
# longer available to new users") on new API keys/projects. Using
# "gemini-flash-latest" instead -- a rolling alias Google maintains that
# always points at the current generally-available Flash model (at time
# of writing, Gemini 3.5 Flash) -- so this won't need updating again
# every time Google renames/retires a specific dated model. If you
# specifically need a pinned, reproducible model version instead of a
# moving alias, replace this with an exact current model ID from
# https://ai.google.dev/gemini-api/docs/models.
# CHANGED AGAIN (2026-07): "gemini-flash-latest" now resolves to
# gemini-3.6-flash, whose FREE TIER daily quota is only 20 requests/day
# (confirmed directly from a real 429 RESOURCE_EXHAUSTED response:
# quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
# quotaValue 20) -- Google's newest models consistently get the
# tightest free-tier caps at launch. Switched to an explicit, older,
# well-established model instead of a rolling alias, specifically for
# its free-tier daily quota: gemini-2.5-flash-lite currently has the
# most generous free RPD of any Gemini model (roughly 1,000-1,500
# requests/day per multiple current sources), which is what a
# 600+-claim batch run actually needs. This is a simple structured JSON
# task (not deep multi-step reasoning), so Flash-Lite's lighter
# capability should be sufficient -- if verification QUALITY (not
# quota) becomes a concern, "gemini-2.5-flash" is the next step up,
# with a smaller but still much larger than 20/day free quota.
# Re-check current limits at https://ai.google.dev/gemini-api/docs/rate-limits
# before assuming these numbers still hold -- free-tier quotas change often.
# CHANGED AGAIN (2026-07): gemini-2.5-flash-lite ALSO returned 404 "no
# longer available to new users" -- turns out the ENTIRE Gemini 2.5
# generation is blocked for new API keys/projects now, not just a
# quota issue on gemini-3.6-flash. Confirmed via a July 2026 upstream
# bug report (googleapis) showing the identical 404 for gemini-2.5-pro
# on new keys, with Google's own error message pointing new users
# specifically to "gemini-3.5-flash, gemini-3.1-flash-lite, or latest
# models." So the 3.x generation is the only one actually available to
# this API key -- the choice is WHICH 3.x model, not whether to use 3.x.
# Picked gemini-3.1-flash-lite over gemini-3.6-flash (20/day quota) or
# gemini-3.5-flash specifically because: (a) it's confirmed available
# to new users per the above, and (b) "flash-lite" tiers consistently
# have the most generous free daily quotas within any given model
# generation, and (c) it's an OLDER, more established release within
# the 3.x family rather than the newest one -- avoiding the "brand new
# model gets the tightest free-tier cap" pattern seen twice now with
# 3.6-flash. Exact current daily quota for 3.1-flash-lite wasn't
# independently confirmed -- watch the first batch of claims after
# switching; if it also hits a low daily cap, gemini-3.5-flash is the
# next thing to try. Re-check https://ai.google.dev/gemini-api/docs/rate-limits
# for current numbers, as these shift often.
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ADDED (2026-07): hard timeout (seconds) for every individual Gemini
# API call in verification.py. google-genai has NO default timeout and
# fully delegates to the server -- there are multiple currently-open
# upstream bug reports of requests hanging indefinitely with zero error
# output ("terminal goes blank"). 60s is generous for a short JSON
# response; lower it if you'd rather fail fast and retry sooner.
GEMINI_TIMEOUT_SECONDS = 60
LLM_MAX_TOKENS_QUERY_GEN = 300
# CORRECTED (this comment was wrong): OPENAI_MODEL_VERIFICATION and
# OPENAI_API_KEY (above) date from a 2026-07-21 attempt to swap the
# verification stage to OpenAI's GPT-4o-mini. That attempt was reverted
# -- src/verification.py imports `google.genai`, never `openai`, and
# DOES still read GEMINI_MODEL/GEMINI_API_KEY directly (verified: grep
# the file). Neither constant below is read by any code path in this
# repo any more; kept only for reference. See root README.md, Notes -> "Backend
# history (Subtask 2)", for the full story.
OPENAI_MODEL_VERIFICATION = "gpt-4o-mini"

# Standalone binary SUPPORTS/REFUTES classifier -- the SAME fine-tuned
# MuRIL checkpoint reused as-is from subtask1_pipeline (identical
# notebook/, identical classifier/config.json -- verified byte-for-byte
# identical between the two repos). Not trained on any Subtask 2 data;
# see root README.md, Notes -> "Classifier". Loaded via
# src/muril_classifier.py::VeracityPredictor, used by
# src/run_classifier_pipeline.py -- a separate, standalone prediction
# path from the Gemini-based run_pipeline.py, not merged into it.
MURIL_CLASSIFIER_MODEL = os.environ.get("MURIL_CLASSIFIER_MODEL", "nirnit-13/muril-supports-refutes")

# ----------------------------------------------------------------------
# Retrieval settings
# ----------------------------------------------------------------------
TOP_K_PER_QUERY = 6       # results pulled per individual search query
SEARCH_TIMEOUT_SECONDS = 15

# ----------------------------------------------------------------------
# NER settings (entity extraction — used for both claim preprocessing
# and evidence snippet preprocessing, feeding the future entity-graph
# ranking step)
# ----------------------------------------------------------------------
# A multilingual NER model. If this fails to load (no internet, model
# name changed, etc.), entity_extraction.py automatically falls back to
# a heuristic extractor so the pipeline degrades instead of crashing.
NER_MODEL = os.environ.get("NER_MODEL", "Davlan/xlm-roberta-base-ner-hrl")

# ----------------------------------------------------------------------
# Output locations
# ----------------------------------------------------------------------
CLAIMS_OUTPUT_DIR = OUTPUT_DIR / "claims"          # one clean JSON per claim
RANKING_SCORES_CSV = OUTPUT_DIR / "ranking_scores.csv"  # tabular BM25/entity/fusion scores, ALL claims
FAILED_CLAIMS_LOG = OUTPUT_DIR / "failed_claims.log"    # claim IDs that errored out, for retry

# ----------------------------------------------------------------------
# Batch run settings
# ----------------------------------------------------------------------
# Small delay between claims to stay under free-tier API rate limits
# when processing many claims in a row. Set to 0 to disable.
API_CALL_DELAY_SECONDS = float(os.environ.get("API_CALL_DELAY_SECONDS", "1.0"))

TOP_K_FINAL = 4

# ----------------------------------------------------------------------
# Neo4j (Entity Graph stage)
# ----------------------------------------------------------------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
# None = server default database. Set NEO4J_DATABASE to e.g. "testkg" to
# target an isolated database on the same server for smoke-testing.
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE") or None

ENTITY_GRAPH_OUTPUT_FILENAME = "entity_ranked.json"
FUSION_OUTPUT_FILENAME = "fusion_ranked.json"

# ----------------------------------------------------------------------
# Final verification stage (fetch full article -> LLM evidence/verdict)
# ----------------------------------------------------------------------
TOP_N_FOR_VERIFICATION = 3            # how many fusion-ranked items to fully fetch + feed the LLM
ARTICLE_FETCH_TIMEOUT_SECONDS = 15
MAX_ARTICLE_CHARS = 6000              # truncate fetched article text before it hits the LLM prompt
# TUNED (2026-07): 4096 -> 8192. Safety margin for the Gemini swap: if
# this model's thinking_config isn't accepted (see verification.py's
# _call_llm_once circuit breaker), internal reasoning may consume part
# of this budget before the actual JSON answer, unlike GPT-4o-mini
# (no hidden reasoning step) or a successfully-thinking-disabled Gemini
# call. Doubling the ceiling gives real headroom either way, at the
# cost of a higher per-call token ceiling (usage-based cost/latency
# impact depends on your plan -- Gemini's free tier counts this too).
LLM_MAX_TOKENS_VERIFICATION = 8192
LLM_TEMPERATURE_VERIFICATION = 0.1   # low temperature -> concise, deterministic, less prone to running long
MAX_VERIFICATION_RETRIES = 2         # total attempts = 1 + this, per FIRE 2026 requirement
VERIFICATION_OUTPUT_FILENAME = "verification_output.json"
FINAL_SUBMISSION_FILENAME = "final_submission.json"       # written directly under OUTPUT_DIR, not per-claim
VALID_PREDICTIONS = ("SUPPORTS", "REFUTES")               # exact, case-sensitive, per FIRE 2026 spec

# ----------------------------------------------------------------------
# Output filenames (STEP 3/4/5 outputs) — centralized here so pipeline.py
# never hardcodes a filename string inline.
# ----------------------------------------------------------------------
RETRIEVAL_OUTPUT_FILENAME = "retrieval_output.json"
BM25_OUTPUT_FILENAME = "bm25_ranked.json"

# NOTE: Fusion-ranking settings (BM25_WEIGHT / ENTITY_WEIGHT /
# CREDIBILITY_WEIGHT_FACTOR / LOW_CREDIBILITY_DOMAINS) have been removed.
# They belonged to the entity-overlap + credibility fusion logic that
# used to live in ranking.py. That logic is deferred to a later stage —
# the current ranker (bm25_ranker.py) is BM25-only. Re-add these here
# when fusion ranking is actually implemented.