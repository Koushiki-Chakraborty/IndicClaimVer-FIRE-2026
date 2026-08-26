import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SUBTASK1_DIR = Path(__file__).parent.parent

CACHE_GROUPS = [
    ("integrated_candidate_cache*.json", "integrated_candidate_cache.merged.json", "candidate"),
    ("integrated_type_cache*.json", "integrated_type_cache.merged.json", "type-flag"),
    ("llm_alias_cache*.json", "llm_alias_cache.merged.json", "LLM-alias"),
]


def _cand_entry_is_fresh(cached: list) -> bool:
    """Same freshness check as integrated_pipeline.py's _cache_entry_is_fresh()
    -- a pre-fix entry missing 'label'/'is_org' should lose to a fresher one
    covering the same key, not overwrite it."""
    return not cached or all("label" in c and "is_org" in c for c in cached)


def merge_candidate_cache(paths: list) -> dict:
    merged = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in raw.items():
            if key not in merged or (_cand_entry_is_fresh(value) and not _cand_entry_is_fresh(merged[key])):
                merged[key] = value
    return merged


def merge_simple(paths: list) -> dict:
    """Type-flag and LLM-alias caches are plain, deterministic key->value
    maps -- any file's value for a given key is as good as any other's,
    so a straight union (last file wins on the rare exact-duplicate key)
    is enough."""
    merged = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def main():
    for pattern, out_name, label in CACHE_GROUPS:
        paths = [p for p in glob.glob(str(SUBTASK1_DIR / pattern)) if not p.endswith(".merged.json")]
        if not paths:
            print(f"[merge] No {label} cache files found matching {pattern} -- skipping.")
            continue

        print(f"[merge] Merging {len(paths)} {label} cache file(s): {[Path(p).name for p in paths]}")
        if "candidate" in out_name:
            merged = merge_candidate_cache(paths)
        else:
            merged = merge_simple(paths)

        out_path = SUBTASK1_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        print(f"[merge] Wrote {len(merged)} entries -> {out_path}\n")


if __name__ == "__main__":
    main()
