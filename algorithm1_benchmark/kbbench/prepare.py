"""Project the frozen release into five dimensions; freeze the query cohort first."""
import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from .common import ROOT, read_json, read_rows, sha256, write_json, write_rows

SEED = 20260930
ANCHORS = [1, 6, 20, 21, 30, 1535, 5114, 9253, 11061, 16498,
           30276, 32281, 40028, 37510, 33352, 20583, 1575, 2001, 199, 431]


def split_for(anime_id):
    digest = hashlib.sha256(f"{SEED}:{anime_id}".encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 10 < 3 else "test"


def prepare(release):
    release = Path(release)
    index = read_json(release / "DATASET_INDEX.json")
    if index["catalog_status"] != "final":
        raise ValueError("Use a frozen final release.")
    source = release / "data/canonical/anime_runtime.jsonl"
    items, relations, counts = [], [], Counter()
    popularity = {}
    for row in read_rows(source):
        if not row["is_runtime_eligible"]:
            raise ValueError("Ineligible item in runtime catalog")
        for refs in row["field_provenance"].values():
            if any(ref.get("source") == "ai_inferred" for ref in refs):
                raise ValueError("Unverified inferred metadata in runtime input")
        demos = sorted(set(row.get("demographics") or []))
        studios = row.get("studios") or []
        # Source arrays preserve declared order. Do not split names containing commas.
        studio = studios[0]["name"].strip() if studios else None
        episodes = row.get("episodes")
        episodes = int(episodes) if episodes is not None and float(episodes) > 0 else None
        item = dict(anime_id=row["anime_id"], title=row["title"],
                    genres=sorted(set(row.get("genres") or [])),
                    themes=sorted(set(row.get("themes") or [])),
                    demographic=demos[0] if demos else None, studio=studio, episodes=episodes)
        items.append(item)
        popularity[row["anime_id"]] = int(row.get("members_count") or 0)
        counts["multi_demographic"] += len(demos) > 1
        counts["multi_studio"] += len(studios) > 1
        for field in ("genres", "themes", "demographic", "studio", "episodes"):
            counts[f"missing_or_empty_{field}"] += not bool(item[field])
        for relation in row.get("relations") or []:
            if relation["relation_type"].casefold() in {"prequel", "sequel"}:
                relations.append(dict(target_id=row["anime_id"], similar_id=relation["target_anime_id"],
                                      relation=relation["relation_type"],
                                      source_refs=row["field_provenance"].get("relations", [])))
    items.sort(key=lambda item: item["anime_id"])
    ids = {item["anime_id"] for item in items}
    assert len(ids) == len(items) == index["counts"]["catalog"]
    relations = [row for row in relations if row["similar_id"] in ids and row["target_id"] != row["similar_id"]]
    write_rows(ROOT / "data/catalog.jsonl.gz", items)
    write_rows(ROOT / "data/franchise_links.jsonl.gz", relations)
    # Deliberately include popular demonstration titles AND report the random stratum separately.
    anchors = [aid for aid in ANCHORS if aid in ids]
    rest = sorted(ids - set(anchors), key=lambda aid: (popularity[aid], aid))
    rng = np.random.default_rng(SEED)
    cohort = [dict(anime_id=aid, sampling_group="anchor", split=split_for(aid)) for aid in anchors]
    for quartile, group in enumerate(np.array_split(rest, 4)):
        selected = sorted(int(aid) for aid in rng.choice(group, size=min(25, len(group)), replace=False))
        cohort.extend(dict(anime_id=aid, sampling_group=f"popularity_q{quartile + 1}", split=split_for(aid)) for aid in selected)
    write_rows(ROOT / "data/targets.jsonl", cohort)
    title_by_id = {item["anime_id"]: item["title"] for item in items}
    with (ROOT / "data/human_annotations_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["target_id", "target_title", "similar_ids", "annotator", "rationale", "status"])
        writer.writeheader()
        for row in cohort:
            writer.writerow(dict(target_id=row["anime_id"], target_title=title_by_id[row["anime_id"]], status="pending"))
    write_json(ROOT / "data/provenance.json", dict(
        catalog_version=index["catalog_version"], catalog_count=len(items), seed=SEED,
        source_files={path.relative_to(release).as_posix(): sha256(path) for path in [source, release / "DATASET_INDEX.json", release / "PACKAGE_MANIFEST.json"]},
        source_selection="C-drive final release supersedes E-drive provisional B handoff; original inputs unchanged",
        projection={"genre_theme": "deduplicated sorted sets", "demographic": "first lexicographic non-empty label",
                    "studio": "first declared studio name, exact casefolded match", "length": "episode count",
                    "empty_labels": "similarity 0 even if both empty", "missing_length": "0.5 as in repository baseline",
                    "semantic_text": "five dimensions only; no title, synopsis, ratings or relations"},
        audit=dict(counts), vocabulary_sizes={field: len({value for item in items for value in item[field]}) for field in ("genres", "themes")},
        target_count=len(cohort), sampling="20 fixed demo anchors + 25 seeded random titles from each membership quartile",
        ground_truth="MAL recommendation votes, independent of co-high-rating training pairs; human labels initially absent"))
    print(f"Prepared {len(items)} titles, {len(cohort)} targets, {len(relations)} direct franchise links", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    prepare(parser.parse_args().release)
