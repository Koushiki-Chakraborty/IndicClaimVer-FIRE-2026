"""
STEP 0: Load raw claims.
STEP 1: Attach a language tag.

The language tag comes for free from WHICH FILE a claim came from — no
language detection model needed. This tag drives every downstream
decision: how many/which-language queries to generate, whether
translation is needed, etc.
"""
import json
from pathlib import Path

from . import config


def load_claims(language: str) -> list:
    """Load all claims for one language, tag each with that language.

    Returns list of dicts: {"id": ..., "text": ..., "lang": ...}
    """
    path = config.DATA_FILES[language]
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"Put your subtask2_{language}_test.json file in {config.DATA_DIR}"
        )

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    claims = []
    for item in raw:
        claims.append({
            "id": item["ID"],
            "text": item["Text"],
            "lang": language,   # <-- STEP 1: the language tag
        })
    return claims


def load_all_claims() -> list:
    """Load and tag claims from all 4 language files. Returns one flat list."""
    all_claims = []
    for lang in config.DATA_FILES:
        try:
            claims = load_claims(lang)
            print(f"[data_loader] Loaded {len(claims)} claims for '{lang}'")
            all_claims.extend(claims)
        except FileNotFoundError as e:
            print(f"[data_loader] WARNING: {e}")
    print(f"[data_loader] Total claims loaded: {len(all_claims)}")
    return all_claims


def get_claim_by_id(claim_id: str) -> dict | None:
    """Convenience: fetch a single claim by ID, searching across all files.
    Useful for the kind of one-claim testing you're doing right now."""
    for lang in config.DATA_FILES:
        try:
            claims = load_claims(lang)
        except FileNotFoundError:
            continue
        for c in claims:
            if c["id"] == claim_id:
                return c
    return None


def select_claims(claim_ids: list | None = None, language: str | None = None,
                   limit: int | None = None) -> list:
    """Single dynamic entry point for deciding WHICH claims a pipeline run
    should process. No claim ID, filename, or language is ever hardcoded
    upstream of this function -- callers pass in whatever selection they
    want at run time (e.g. from CLI flags), and this is the only place
    that turns that selection into an actual list of claim dicts.

    - claim_ids given  -> fetch exactly those claims (order preserved;
                           missing IDs are skipped with a warning, not
                           a crash, so one bad ID doesn't kill a batch run)
    - language given    -> every claim from that one language file
    - neither given      -> every claim across ALL language files
    - limit (optional)   -> cap the resulting list, applied last

    This one function covers all three cases you described: today
    testing 1 claim, tomorrow running 1901 claims -- same function,
    different arguments, zero code changes in between.
    """
    if claim_ids:
        claims = []
        for cid in claim_ids:
            claim = get_claim_by_id(cid)
            if claim is None:
                print(f"[data_loader] WARNING: claim ID not found, skipping: {cid}")
                continue
            claims.append(claim)
    elif language:
        claims = load_claims(language)
    else:
        claims = load_all_claims()

    if limit is not None:
        claims = claims[:limit]

    return claims


def load_claims_from_file(path: str, lang: str = "dev") -> list:
    """OPTIONAL ADD-ON: load claims from an ARBITRARY JSON file path,
    bypassing config.DATA_FILES entirely -- e.g. for a dev/held-out
    dataset like dev_subtask2.json that isn't one of the 4 test
    language files. Same {"ID","Text"} input schema, same output shape
    as load_claims() -> {"id", "text", "lang"}, so every downstream
    stage (query generation, retrieval, ranking, verification) works
    completely unchanged on claims loaded this way.

    `lang` tags every claim the same way load_claims() tags claims by
    filename -- pass whatever is appropriate for your dev file (default
    "dev" if the file doesn't correspond to one specific language).
    Extra keys on each claim record (e.g. a gold "Label" field used
    later for your own F1 computation) are simply ignored here, exactly
    like load_claims() already ignores anything beyond "ID"/"Text".

    Does not modify load_claims(), load_all_claims(), select_claims(),
    or get_claim_by_id() -- this is a new, independent entry point.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)

    claims = []
    for item in raw:
        claims.append({
            "id": item["ID"],
            "text": item["Text"],
            "lang": lang,
        })
    return claims


if __name__ == "__main__":
    # Quick smoke test — run this file directly to check your data files load correctly
    claims = load_all_claims()
    if claims:
        print("\nSample claim:")
        print(claims[0])