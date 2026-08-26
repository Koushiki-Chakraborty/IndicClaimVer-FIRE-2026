import json
import re
from collections import defaultdict
from pathlib import Path

import config


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all():
    """Returns (train, dev, pool) -- any missing file returns []
    with a warning rather than crashing, so partial pipelines still run."""
    out = {}
    for name, path in [
        ("train", config.TRAIN_PATH),
        ("dev", config.DEV_PATH),
        ("pool", config.POOL_PATH),
    ]:
        if path.exists():
            out[name] = load_json(path)
        else:
            print(f"[data_loader] WARNING: {path} not found. "
                  f"Place the official task file there. Returning [].")
            out[name] = []
    return out["train"], out["dev"], out["pool"]


def detect_lang(text: str) -> str:
    """Character-count-weighted script detection. Indic scripts weighted
    2x because they are unambiguous signals; a single Latin word inside
    an otherwise-Devanagari sentence should not flip the language."""
    bn = len(re.findall(r"[\u0980-\u09FF]", text))
    hi = len(re.findall(r"[\u0900-\u097F]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    scores = {"bn": bn * 2, "hi": hi * 2, "en": en}
    return max(scores, key=scores.get)


def is_code_mixed(text: str) -> bool:
    has_indic = bool(re.search(r"[\u0900-\u09FF]", text))
    has_latin = bool(re.search(r"[a-zA-Z]{4,}", text))
    return has_indic and has_latin


def build_pool_text_to_id(pool: list) -> dict:
    return {item["Evidence"]: item["ID"] for item in pool}


def build_pool_by_lang(pool: list) -> dict:
    by_lang = defaultdict(list)
    for item in pool:
        by_lang[detect_lang(item["Evidence"])].append(item)
    return dict(by_lang)


if __name__ == "__main__":
    train, dev, pool = load_all()
    print(f"train={len(train)} dev={len(dev)} pool={len(pool)}")
    if train:
        langs = defaultdict(int)
        for item in train:
            langs[detect_lang(item["Text"])] += 1
        print("Train language distribution:", dict(langs))
        codemix = sum(1 for item in train if is_code_mixed(item["Text"]))
        print(f"Code-mixed claims: {codemix}")
