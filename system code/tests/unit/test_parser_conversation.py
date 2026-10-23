import pytest

from app.chat.conversation import effective_intent, merge
from app.chat.intent import TagLexicon, parse
from app.retrieval.titles import TitleResolver
from tests.helpers import make_snapshot


@pytest.fixture(scope="module")
def world():
    snap = make_snapshot([
        {"title": "Crimson Orbit", "genre": ["Action", "Sci-Fi"], "theme": ["Mecha"], "episodes": 24},
        {"title": "Quiet Garden", "genre": ["Slice of Life", "Comedy"], "theme": ["Iyashikei"], "episodes": 12},
        {"title": "Hollow Signal", "genre": ["Mystery", "Horror"], "theme": ["Psychological"], "episodes": 13},
        {"title": "Monster", "genre": ["Mystery"], "episodes": 74},
    ])
    aliases = [(k, t, "title") for k, t in zip(snap.keys, snap.titles, strict=True)]
    titles = TitleResolver(aliases, dict(zip(snap.keys, snap.titles, strict=True)), {}, set(snap.keys))
    return snap, titles, TagLexicon(snap)


def P(world, text):
    snap, titles, lex = world
    return parse(text, snap, titles, lex)


def test_numeric_and_negation(world):
    pq = P(world, "a comedy anime with at most 12 episodes and no horror, score at least 7.5")
    assert pq.max_episodes == 12 and pq.min_score == 7.5
    assert ["genre", "Comedy"] in pq.include_tags and ["genre", "Horror"] in pq.exclude_tags
    assert pq.intent == "preference_recommendation"
    assert P(world, "under 12 episodes").max_episodes == 11          # exclusive bound


def test_soft_markers_and_vague_length_stay_soft(world):
    pq = P(world, "something like Crimson Orbit but not too long, ideally with some romance")
    assert pq.intent == "similar_to" and pq.reference_titles[0]["title"] == "Crimson Orbit"
    assert pq.max_episodes is None                                    # never silently 12 episodes
    assert "short" in pq.soft_preferences


def test_shorter_than_reference(world):
    pq = P(world, "like Crimson Orbit but shorter")
    assert "shorter_than_reference" in pq.soft_preferences and pq.max_episodes is None


def test_typo_title_and_relation_and_plan_intents(world):
    assert P(world, "tell me about Crimsn Orbit").reference_titles[0]["title"] == "Crimson Orbit"
    assert P(world, "what is the sequel to Hollow Signal?").intent == "relation_query"
    pq = P(world, "I have 15 hours this weekend, at most 3 titles, different styles")
    assert pq.intent == "viewing_plan" and pq.budget_minutes == 900 and pq.max_titles == 3 and pq.diversity == "high"
    assert P(world, "which episode has the rooftop scene in Hollow Signal").intent == "scene_lookup"
    assert P(world, "what's the weather today").intent == "out_of_scope"


def test_lowercase_common_word_is_not_a_title(world):
    pq = P(world, "a monster hunting show with lots of fighting")
    assert not pq.reference_titles                                    # "monster" != the title Monster
    assert ["genre", "Action"] in pq.include_tags


def test_contradiction_needs_clarification(world):
    pq = P(world, "at least 50 episodes but at most 3 episodes")
    assert pq.needs_clarification and "50" in pq.clarification


def test_chinese_numbers_and_tags(world):
    pq = P(world, "想看搞笑的，最多12集，不要恐怖，评分7.5分以上")
    assert pq.max_episodes == 12 and pq.min_score == 7.5
    assert ["genre", "Comedy"] in pq.include_tags and ["genre", "Horror"] in pq.exclude_tags


def test_multi_turn_refinement_keeps_reference(world):
    st, _, refine = merge(None, P(world, "something like Crimson Orbit"), "something like Crimson Orbit")
    st["last_intent"] = "similar_to"
    pq2 = P(world, "at most 13 episodes")
    st2, changes, refine2 = merge(st, pq2, "at most 13 episodes")
    assert refine2 and st2["reference"]["title"] == "Crimson Orbit" and st2["max_episodes"] == 13
    assert effective_intent(pq2, st2, refine2) == "similar_to"
    assert any("max_episodes" in c for c in changes)
    pq3 = P(world, "a relaxing slice of life about cooking and friendship")
    st3, _, refine3 = merge(st2, pq3, "a relaxing slice of life about cooking and friendship")
    assert not refine3 and st3["reference"] is None and st3["max_episodes"] is None   # new request -> fresh state


def test_sequels_differing_only_in_punctuation_resolve_to_the_literal_title():
    from app.retrieval.titles import TitleResolver

    aliases = [("mal:6956", "Working!!", "title"), ("mal:10521", "Working'!!", "title"),
               ("mal:37141", "Hataraku Saibou", "title"), ("mal:39586", "Hataraku Saibou!!", "title")]
    titles = {k: a for k, a, _ in aliases}
    popularity = {"mal:6956": 10.0, "mal:10521": 5.0, "mal:37141": 10.0, "mal:39586": 5.0}   # popular one first
    r = TitleResolver(aliases, titles, popularity, set(titles))
    assert r.resolve("Working'!!")[0].item_key == "mal:10521"          # the literal sequel title wins
    assert r.resolve("Working!!")[0].item_key == "mal:6956"
    assert r.resolve("working")[0].item_key == "mal:6956"             # no literal match: popularity decides
    assert r.resolve("Hataraku Saibou!!")[0].item_key == "mal:39586"
    found = r.find_mentions("is Hataraku Saibou!! worth it")
    assert found and found[0][0].item_key == "mal:39586"


def test_cue_references_keep_punctuation_that_belongs_to_the_title():
    from app.chat.intent import TagLexicon, parse
    from app.retrieval.titles import TitleResolver
    from tests.helpers import make_snapshot

    rows = [{"item_key": "mal:6956", "title": "Working!!"}, {"item_key": "mal:10521", "title": "Working'!!"},
            {"item_key": "mal:50631", "title": "Komi-san wa, Comyushou desu. 2nd Season"},
            {"item_key": "mal:482", "title": "Yu-Gi-Oh! GX"}]
    snap = make_snapshot(rows)
    aliases = [(r["item_key"], r["title"], "title") for r in rows]
    titles = TitleResolver(aliases, {r["item_key"]: r["title"] for r in rows}, {"mal:6956": 9.0}, set(snap.keys))
    lex = TagLexicon(snap)
    for msg, key in (("tell me about Working'!!", "mal:10521"),
                     ("tell me about Komi-san wa, Comyushou desu. 2nd Season", "mal:50631"),
                     ("what is Yu-Gi-Oh! GX?", "mal:482"),
                     ("something like Working!! but shorter", "mal:6956")):
        pq = parse(msg, snap, titles, lex)
        assert pq.reference_titles and pq.reference_titles[0]["item_key"] == key, msg


def test_typo_in_a_title_with_punctuation_still_resolves():
    from app.chat.intent import TagLexicon, parse
    from app.retrieval.titles import TitleResolver
    from tests.helpers import make_snapshot

    rows = [{"item_key": "mal:50631", "title": "Komi-san wa, Comyushou desu. 2nd Season"},
            {"item_key": "mal:369", "title": "Boogiepop wa Warawanai"}]
    snap = make_snapshot(rows)
    titles = TitleResolver([(r["item_key"], r["title"], "title") for r in rows],
                           {r["item_key"]: r["title"] for r in rows}, {}, set(snap.keys))
    pq = parse("tell me about Komi-san wa, Cmyushou desu. 2nd Season", snap, titles, TagLexicon(snap))
    assert pq.reference_titles and pq.reference_titles[0]["item_key"] == "mal:50631"
