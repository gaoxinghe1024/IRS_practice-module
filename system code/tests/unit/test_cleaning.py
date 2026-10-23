import pytest

from pipelines import cleaning as C


@pytest.mark.parametrize("raw, minutes, per_ep", [
    ("24 min per ep", 24.0, True), ("1 hr 30 min", 90.0, False), ("2 hr", 120.0, False),
    ("45 sec", 0.75, False), ("30 sec per ep", 0.5, True), ("1 hr 5 min per ep", 65.0, True),
    ("Unknown", None, False), ("", None, False)])
def test_parse_duration(raw, minutes, per_ep):
    d = C.parse_duration(raw)
    assert d.minutes == minutes
    if minutes is not None:
        assert d.per_episode_explicit is per_ep


def test_runtime_never_imputed():
    # unknown episodes with a per-episode duration: per-episode known, total unknown
    assert C.resolve_runtime(C.parse_duration("24 min per ep"), None) == (24.0, None, "per_episode_average")
    # multi-episode title without "per ep": ambiguous -> unknown (never 24 x N)
    assert C.resolve_runtime(C.parse_duration("24 min"), 12) == (None, None, None)
    assert C.resolve_runtime(C.parse_duration("1 hr 30 min"), 1) == (90.0, 90.0, "total_runtime")
    assert C.resolve_runtime(C.parse_duration("Unknown"), 12) == (None, None, None)


def test_multi_value_keeps_corporate_suffixes_and_dedupes():
    assert C.parse_multi("Yamamura Animation, Inc., NHK, NHK") == ["Yamamura Animation, Inc.", "NHK"]
    assert C.parse_multi("UNKNOWN") == []
    assert C.parse_multi("Action, Award Winning, Sci-Fi") == ["Action", "Award Winning", "Sci-Fi"]


@pytest.mark.parametrize("raw, start, end", [
    ("Apr 3, 1998 to Apr 24, 1999", "1998-04-03", "1999-04-24"), ("Apr 3, 1998", "1998-04-03", None),
    ("1998", "1998", None), ("Apr 1998 to ?", "1998-04", None), ("2020 to 2021", "2020", "2021"),
    ("Not available", None, None), ("Mar 5, 2010 to 2011", "2010-03-05", "2011")])
def test_parse_aired(raw, start, end):
    assert C.parse_aired(raw) == (start, end)


def test_premiered_and_numbers():
    assert C.parse_premiered("spring 1998") == ("spring", 1998)
    assert C.parse_premiered("UNKNOWN") == (None, None)
    assert C.parse_int("26.0") == 26 and C.parse_int("1,234") == 1234
    assert C.parse_int("UNKNOWN") is None and C.parse_int("2.5") is None and C.parse_int("-3") is None
    assert C.parse_float("8.75", 1, 10) == 8.75 and C.parse_float("UNKNOWN") is None
    assert C.parse_float("11", 1, 10) is None


def test_placeholder_synopsis_is_missing():
    assert C.parse_synopsis("No description available for this anime.") is None
    assert C.parse_synopsis("  A real synopsis. ") == "A real synopsis."


def test_explicit_content_rule():
    rules = (["Hentai", "Erotica"], ["Rx"])
    assert C.is_explicit(["Hentai"], "UNKNOWN", *rules)
    assert C.is_explicit(["Comedy"], "Rx - Hentai", *rules)
    assert not C.is_explicit(["Ecchi", "Comedy"], "R+ - Mild Nudity", *rules)   # Ecchi / R+ are not explicit
    assert not C.is_explicit(["Drama"], "R - 17+ (violence & profanity)", *rules)


def test_jikan_provenance_counts_only_endpoints_that_returned_data():
    from pipelines.real_ingest import _jikan_provenance

    assert _jikan_provenance(None) == {"jikan_enriched": False, "jikan_endpoints_used": [], "jikan_fetched_at": None}
    rec = {"mal_id": 1, "fetched_at": "2026-09-30T12:15:49Z", "endpoints": {
        "full": {"status": 200, "fetched_at": "2026-09-30T10:46:59Z", "data": {"data": {"mal_id": 1}}},
        "recommendations": {"status": 504, "fetched_at": "2026-09-30T12:15:49Z", "data": None},
        "characters": {"status": 504, "fetched_at": "2026-09-30T12:15:49Z", "data": None}}}
    assert _jikan_provenance(rec) == {"jikan_enriched": True, "jikan_endpoints_used": ["full"],
                                      "jikan_fetched_at": "2026-09-30T10:46:59Z"}
    failed = {"mal_id": 2, "endpoints": {"full": {"status": 504, "data": None}}}
    assert _jikan_provenance(failed)["jikan_enriched"] is False       # a cache file alone is not enrichment


def test_supplementary_files_are_validated_and_mapped_onto_processed_tables(tmp_path):
    import hashlib
    import json

    import pytest

    from app.core.config import load_settings
    from pipelines.real_ingest import DataValidationError, load_supplementary

    enr = tmp_path / "data" / "enriched"
    (enr / "scene_images" / "frames").mkdir(parents=True)
    img = enr / "scene_images" / "frames" / "f1.png"
    img.write_bytes(b"fixture bytes")

    def write(name, rows):
        (enr / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    write("relations.jsonl", [
        {"from_mal_id": 1, "to_mal_id": 2, "relation_type": "Sequel", "source_ref": "curated:1"},
        {"from_mal_id": 1, "to_external_id": "manga:5", "to_entity_type": "manga", "to_name": "Source",
         "relation_type": "adaptation", "source_ref": "curated:2"}])
    write("characters.jsonl", [{"mal_id": 1, "character_source_id": "c1", "name": "A", "source_ref": "curated:3"},
                               {"mal_id": 99, "character_source_id": "c2", "name": "B", "source_ref": "curated:4"}])
    write("scenes.jsonl", [{"scene_id": "s1", "item_key": "mal:1", "description": "rooftop talk", "episode": 3,
                            "source_ref": "curated:5", "verification_status": "verified"}])
    write("scene_images/manifest.jsonl", [{"image_id": "i1", "item_key": "mal:1", "scene_id": "s1",
                                           "path": "frames/f1.png", "source_ref": "curated:6",
                                           "sha256": hashlib.sha256(img.read_bytes()).hexdigest()}])
    s = load_settings("real", environ={"ANILOGIC_PATHS__DATA_ROOT": str(tmp_path / "data"),
                                       "ANILOGIC_REAL__ENRICHED_DIR": str(enr)})
    out = load_supplementary(s, {"mal:1", "mal:2"})
    assert [r["relation_type"] for r in out["relations"]] == ["sequel", "adaptation"]
    assert out["relations"][1]["to_type"] == "manga" and out["relations"][1]["to_mal"] == "manga:5"
    assert [c["item_key"] for c in out["characters"]] == ["mal:1"]
    assert out["report"]["characters.jsonl"]["outside_catalogue"] == 1          # counted, not silently kept
    assert out["scenes"][0]["episode"] == 3 and out["scenes"][0]["spoiler_level"] == "major"
    assert out["scene_images"][0]["path"] == "data/enriched/scene_images/frames/f1.png"
    write("scenes.jsonl", [{"scene_id": "s2", "item_key": "mal:1", "description": "d", "source_ref": "x",
                            "verification_status": "probably"}])
    with pytest.raises(DataValidationError):                                    # invalid file: nothing imported
        load_supplementary(s, {"mal:1"})
