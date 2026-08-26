import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from subtask1 import entity_linker, data_loader

with open("data_test/evidence_pool.json", encoding="utf-8") as f:
    pool = json.load(f)

raw_ner = entity_linker._get_ner()


def old_extract(text):
    results = raw_ner(text[:512])
    mentions = [r["word"].strip() for r in results if len(r["word"].strip()) > 1]
    seen, out = set(), []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def old_link(label, lang):
    return entity_linker._link_entity_query(label, lang)


old_stats = {"with_entity": 0, "zero_entity": 0, "total_linked": 0}
new_stats = {"with_entity": 0, "zero_entity": 0, "total_linked": 0}

for i, item in enumerate(pool):
    text = item["Evidence"]
    lang = data_loader.detect_lang(text)

    old_mentions = old_extract(text)
    old_linked = [old_link(m, lang) for m in old_mentions]
    old_n = sum(1 for q in old_linked if q is not None)
    old_stats["total_linked"] += old_n
    if old_n > 0:
        old_stats["with_entity"] += 1
    else:
        old_stats["zero_entity"] += 1

    new_mentions = entity_linker.extract_entities(text)
    new_linked = [entity_linker.link_entity(m, lang) for m in new_mentions]
    new_n = sum(1 for q in new_linked if q is not None)
    new_stats["total_linked"] += new_n
    if new_n > 0:
        new_stats["with_entity"] += 1
    else:
        new_stats["zero_entity"] += 1

    print(f"[{i+1}/{len(pool)}] {item['ID']} ({lang}): old={old_n} new={new_n}", flush=True)

n = len(pool)
print("\n=== RESULTS ===")
print(f"Total evidence items: {n}")
print(f"OLD -- with >=1 entity: {old_stats['with_entity']} ({100*old_stats['with_entity']/n:.1f}%), "
      f"zero entity: {old_stats['zero_entity']} ({100*old_stats['zero_entity']/n:.1f}%), "
      f"avg linked/item: {old_stats['total_linked']/n:.2f}")
print(f"NEW -- with >=1 entity: {new_stats['with_entity']} ({100*new_stats['with_entity']/n:.1f}%), "
      f"zero entity: {new_stats['zero_entity']} ({100*new_stats['zero_entity']/n:.1f}%), "
      f"avg linked/item: {new_stats['total_linked']/n:.2f}")
