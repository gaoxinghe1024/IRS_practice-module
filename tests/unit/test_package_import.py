"""Team data-package converter: field mapping, policies (no core-field imports, one demographic, privacy),
evaluation-query mapping and overwrite protection."""

import builtins
import gzip
import json
import pathlib

import pytest

from app.core.config import load_settings
from pipelines.package_import import PackageError, import_package, import_similarity_labels
from pipelines.validate_data import CONTRACTS, validate_file


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def package(tmp_path):
    pkg = tmp_path / "Data"
    (pkg / "data" / "canonical").mkdir(parents=True)
    (pkg / "DATASET_INDEX.json").write_text(json.dumps({"catalog_status": "final", "catalog_version": "t1",
                                                        "schema_version": 1}), encoding="utf-8")
    prov = {"themes": [{"source": "anilist_detail"}], "demographics": [{"source": "anime2023"}],
            "synopsis_en": [{"source": "anime2023"}], "relations": [{"source": "jikan_relations"}]}
    _jsonl(pkg / "data" / "canonical" / "anime_all.jsonl", [
        {"anime_id": 1, "identity_status": "confirmed", "genres": ["Action", "Space Opera"], "studios": ["Sunrise"],
         "score": 9.9, "episodes": 99, "themes": ["Space", "Space", "Adult Cast"], "demographics": ["Shounen", "Kids"],
         "synopsis_en": "snapshot text", "field_provenance": json.dumps(prov),
         "relations": json.dumps([{"target_anime_id": 2, "relation_type": "Side Story"},
                                  {"target_anime_id": 2, "relation_type": "Side Story"}]),
         "characters": [{"name": "Spike", "role": "Main", "source_character_id": 7, "entity_namespace": "mal"}]},
        {"anime_id": 2, "identity_status": "confirmed", "themes": [], "demographics": [],
         "synopsis_en": "A filled synopsis.", "field_provenance": {"synopsis_en": [{"source": "external_verified"}]}},
        {"anime_id": 3, "identity_status": "unresolved", "themes": ["Music"]},
    ])
    _jsonl(pkg / "data" / "rag" / "scenes_verified.jsonl", [
        {"scene_id": "scene_A_1", "anime_id": 1, "title_en": "Spike meets Asimov", "description_en": "They talk.",
         "episode": 1, "timestamp_seconds": None, "characters": ["Spike"], "verification_status": "verified",
         "verified_at": "2026-09-18T06:24:52Z", "verified_by": "executor:A (Codex)",
         "verification_method": "source_text", "sources": [{"url": "https://example.org/ep1", "title": "Ep list"}]},
    ])
    _jsonl(pkg / "data" / "rag" / "evaluation_queries.jsonl", [
        {"query_id": "EVAL-001", "category": "scene", "query_en": "Which anime has Spike meet Asimov?",
         "expected_anime_ids": [1], "expected_scene_ids": ["scene_A_1"], "constraints": {},
         "expected_outcome": "matches", "evidence_note": "anchored"},
        {"query_id": "EVAL-074", "category": "constraint", "query_en": "Action since 2010",
         "expected_anime_ids": [1, 2], "expected_scene_ids": [],
         "constraints": {"genres": ["Action"], "year_from": 2010, "type": "TV", "demographics": ["Shounen"]},
         "expected_outcome": "matches"},
        {"query_id": "EVAL-092", "category": "unanswerable", "query_en": "Titles from 2036 or later",
         "expected_anime_ids": [], "expected_scene_ids": [], "constraints": {"year_from": 2036},
         "expected_outcome": "no_match"},
    ])
    with gzip.open(pkg / "data" / "canonical" / "users.csv.gz", "wt") as fh:
        fh.write("user_id,source_username,gender\n1,someone,Female\n")
    s = load_settings("real", environ={"ANILOGIC_PATHS__DATA_ROOT": str(tmp_path / "data"),
                                       "ANILOGIC_REAL__ENRICHED_DIR": str(tmp_path / "data" / "enriched")})
    return pkg, s, tmp_path / "data"


def _read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_converter_maps_fields_and_keeps_policies(package, monkeypatch):
    pkg, s, data = package
    opened = []
    real_open, real_path_open = builtins.open, pathlib.Path.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_path(self, *a, **k):
        opened.append(str(self))
        return real_path_open(self, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(pathlib.Path, "open", spy_path)
    report = import_package(s, pkg, log=lambda *_: None)
    monkeypatch.undo()
    assert not any("users.csv" in p for p in opened)                  # usernames / gender are never read
    tags = _read(data / "enriched" / "tags.jsonl")
    assert {t["namespace"] for t in tags} == {"theme", "demographic"}      # no genre / studio values imported
    assert sorted(t["value"] for t in tags if t["namespace"] == "theme") == ["Adult Cast", "Space"]
    assert [t["value"] for t in tags if t["namespace"] == "demographic"] == ["Kids"]   # one, alphabetically first
    assert report["demographics_reduced"] == [{"mal_id": 1, "package": ["Kids", "Shounen"], "kept": "Kids"}]
    assert all(t["mal_id"] != 3 for t in tags)                          # unresolved identity skipped
    syn = _read(data / "enriched" / "synopses.jsonl")
    assert [(x["mal_id"], x["source"]) for x in syn] == [(2, "external_verified")]   # not the snapshot's text
    rel = _read(data / "enriched" / "relations.jsonl")
    assert rel == [{"from_mal_id": 1, "to_mal_id": 2, "relation_type": "side_story",
                    "source_ref": "team-package:t1:anime_all#1/relations (jikan_relations)"}]
    chars = _read(data / "enriched" / "characters.jsonl")
    assert chars[0]["character_source_id"] == "mal:7" and chars[0]["role"] == "Main"
    scene = _read(data / "enriched" / "scenes.jsonl")[0]
    assert scene["item_key"] == "mal:1" and scene["description"] == "Spike meets Asimov. They talk."
    assert scene["verification_status"] == "verified" and scene["spoiler_level"] == "major"
    assert "checked 2026-09-18 by executor:A (Codex) (source_text)" in scene["source_ref"]
    assert scene["source_url"] == "https://example.org/ep1" and scene["timestamp_seconds"] is None
    for name in ("tags", "synopses", "relations", "characters", "scenes"):
        assert validate_file(data / "enriched" / f"{name}.jsonl", CONTRACTS[f"enriched/{name}.jsonl"])["ok"]


def test_queries_keep_their_label_source_and_outcome(package):
    pkg, s, data = package
    import_package(s, pkg, log=lambda *_: None)
    q = {r["query_id"]: r for r in _read(data / "evaluation" / "rag_queries.jsonl")}
    assert all(r["label_source"] == "team_package_t1" for r in q.values())
    assert q["team-EVAL-001"]["intent"] == "scene_lookup" and q["team-EVAL-001"]["expected_scene_ids"] == ["scene_A_1"]
    hc = q["team-EVAL-074"]["hard_constraints"]
    assert hc == {"include_genres": ["Action"], "year_from": 2010, "types": ["TV"], "include_demographics": ["Shounen"]}
    assert q["team-EVAL-074"]["relevant_item_keys"] == ["mal:1", "mal:2"]
    assert q["team-EVAL-092"]["intent"] is None and q["team-EVAL-092"]["answerable"] is False
    assert q["team-EVAL-092"]["expected_outcome"] == "no_match"
    assert validate_file(data / "evaluation" / "rag_queries.jsonl", CONTRACTS["evaluation/rag_queries.jsonl"])["ok"]


def test_refuses_to_replace_other_labels_and_unfinished_packages(package):
    pkg, s, data = package
    _jsonl(data / "evaluation" / "rag_queries.jsonl", [{"query_id": "h1", "query": "q", "intent": "anime_fact",
                                                        "relevant_item_keys": [], "answerable": True,
                                                        "label_source": "human_panel"}])
    with pytest.raises(PackageError, match="--force"):
        import_package(s, pkg, log=lambda *_: None)
    import_package(s, pkg, force=True, log=lambda *_: None)
    import_package(s, pkg, log=lambda *_: None)                       # re-import over its own output is fine
    idx = json.loads((pkg / "DATASET_INDEX.json").read_text())
    (pkg / "DATASET_INDEX.json").write_text(json.dumps({**idx, "catalog_status": "draft"}))
    with pytest.raises(PackageError, match="final"):
        import_package(s, pkg, log=lambda *_: None)


def test_similarity_labels_keep_the_benchmark_split(package, tmp_path):
    _pkg, s, data = package
    src = tmp_path / "gt.jsonl"
    _jsonl(src, [{"target_id": 1, "split": "historical", "relevant_ids": [2, 3], "votes": [5, 1]},
                 {"target_id": 4, "split": "test", "relevant_ids": [5]},
                 {"target_id": 6, "split": "validation", "relevant_ids": []}])
    rep = import_similarity_labels(s, src, log=lambda *_: None)
    rows = _read(data / "evaluation" / "similarity_labels_mal.jsonl")
    assert [(r["query_item"], r["split"]) for r in rows] == [("mal:1", "train"), ("mal:4", "test")]
    assert rows[0]["positives"][0] == {"item_key": "mal:2", "grade": 1, "votes": 5}
    assert rows[0]["label_source"] == "mal_user_recommendations_weak" and rep["splits"]["skipped"] == 1


def test_catalog_supplements_are_validated_and_keyed_by_mal_id(tmp_path):
    from pipelines.real_ingest import DataValidationError, load_catalog_supplements

    enr = tmp_path / "data" / "enriched"
    _jsonl(enr / "tags.jsonl", [{"mal_id": 1, "namespace": "theme", "value": "Space", "source": "anilist",
                                 "source_ref": "x"},
                                {"mal_id": 1, "namespace": "theme", "value": "Space", "source": "dup", "source_ref": "x"}])
    _jsonl(enr / "synopses.jsonl", [{"mal_id": 2, "synopsis": "Text.", "source": "external", "source_ref": "y"}])
    s = load_settings("real", environ={"ANILOGIC_PATHS__DATA_ROOT": str(tmp_path / "data"),
                                       "ANILOGIC_REAL__ENRICHED_DIR": str(enr)})
    out = load_catalog_supplements(s)
    assert out["tags"] == {1: {"theme": [("Space", "anilist")]}}           # duplicates collapse, first source kept
    assert out["synopses"] == {2: ("Text.", "external")}
    assert out["report"]["tags.jsonl"]["rows"] == 2
    _jsonl(enr / "tags.jsonl", [{"mal_id": 1, "namespace": "mood", "value": "x", "source_ref": "z"}])
    with pytest.raises(DataValidationError):                                # unknown namespace: nothing merged
        load_catalog_supplements(s)


def test_logistic_split_keeps_a_preregistered_split_and_audits_franchise_overlap():
    from app.recommendation.similarity.logistic import group_queries, split_franchise_overlap

    rows = [{"query_item": "mal:1", "split": "train"}, {"query_item": "mal:2", "split": "test"},
            {"query_item": "mal:3", "split": "val"}]
    comps = {"mal:1": "f1", "mal:2": "f1", "mal:3": "f3"}
    groups = group_queries(rows, comps, seed=7)
    assert groups == {"mal:1": "train", "mal:2": "test", "mal:3": "val"}
    audit = split_franchise_overlap(groups, comps)
    assert audit["test"] == {"queries": 1, "sharing_franchise_with_other_split": 1}
    assert audit["val"]["sharing_franchise_with_other_split"] == 0
    unsplit = group_queries([{"query_item": "mal:1"}, {"query_item": "mal:2"}], comps, seed=7)
    assert unsplit["mal:1"] == unsplit["mal:2"]                             # same franchise -> same split


def test_chat_evaluation_uses_real_labels_only_in_real_mode(tmp_path):
    from evaluation.experiments.chat import _labelled, _split_by_category

    env = {"ANILOGIC_PATHS__DATA_ROOT": str(tmp_path / "data")}
    _jsonl(tmp_path / "data" / "evaluation" / "rag_queries.jsonl",
           [{"query_id": "t1", "query": "q", "intent": None, "relevant_item_keys": ["mal:1"], "answerable": True,
             "category": "scene", "label_source": "team_package_t1"}])
    real = _labelled(load_settings("real", environ=env))
    assert real[0] == "team_package_t1" and list(_split_by_category(*real)) == ["team_package_t1:scene"]
    synthetic = load_settings("synthetic", environ=env)
    _jsonl(synthetic.synthetic_dir / "evaluation" / "rag_queries.jsonl",
           [{"query_id": "s1", "query": "q", "intent": "anime_fact", "relevant_item_keys": [], "answerable": True}])
    assert _labelled(synthetic)[0] == "labelled_synthetic_generator"      # MAL-keyed labels never reach synthetic
