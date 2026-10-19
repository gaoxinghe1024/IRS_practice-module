"""Start-year, score-range and studio constraints; requests for data the catalogue does not hold;
"which anime has <plot>" questions (parser, constraint engine, conversation state)."""

import pytest

from app.chat.conversation import constraint_chips, merge
from app.chat.intent import TagLexicon, parse
from app.recommendation.constraints import ConstraintEngine, ConstraintSet
from app.retrieval.titles import TitleResolver
from tests.helpers import make_snapshot


@pytest.fixture(scope="module")
def world():
    snap = make_snapshot([
        {"title": "Old Guard", "genre": ["Action"], "demographic": ["Shounen"], "aired_from": "1998", "score": 8.1,
         "studio": ["Madhouse"]},
        {"title": "New Wave", "genre": ["Action"], "aired_from": "2016", "score": 8.6, "studio": ["Kyoto Animation"]},
        {"title": "Mid Point", "genre": ["Comedy"], "aired_from": "2010", "score": 7.2, "type": "Movie",
         "studio": ["Bones"]},
        {"title": "No Date", "genre": ["Comedy"], "aired_from": None, "score": 9.0},
        {"title": "Adult Only", "genre": ["Hentai"], "is_explicit": True, "aired_from": "2012"},
        {"title": "A-Girl", "genre": ["Romance"], "theme": ["School"], "aired_from": "2014"},
        {"title": "School", "genre": ["Comedy"], "theme": ["School"], "aired_from": "2003"},
    ])
    aliases = [(k, t, "title") for k, t in zip(snap.keys, snap.titles, strict=True)]
    titles = TitleResolver(aliases, dict(zip(snap.keys, snap.titles, strict=True)), {}, set(snap.keys))
    return snap, titles, TagLexicon(snap)


def P(world, text):
    snap, titles, lex = world
    return parse(text, snap, titles, lex)


def years(world, text):
    pq = P(world, text)
    return pq.year_from, pq.year_to


@pytest.mark.parametrize("text,expected", [
    ("Find action anime that began airing in 2010 or later.", (2010, None)),
    ("Which shounen titles began airing in 2000 or earlier?", (None, 2000)),
    ("Show titles that started in 2015 or later and have at most 12 episodes.", (2015, None)),
    ("a comedy since 2015", (2015, None)),
    ("romance after 2010", (2011, None)),            # "after" excludes the year itself
    ("anime before 2000", (None, 1999)),
    ("between 2005 and 2010, comedy", (2005, 2010)),
    ("90s mecha anime", (1990, 1999)),
    ("anime from 2010", (2010, 2010)),
    ("2010年以后的科幻动画", (2010, None)),
    ("2000年以前的动画", (None, 1999)),
    ("2000年及以前的动画", (None, 2000)),
])
def test_year_phrases(world, text, expected):
    assert years(world, text) == expected


def test_a_year_inside_a_plot_or_a_count_is_not_a_limit(world):
    pq = P(world, "Which film follows an alchemist stranded on a parallel Earth in 1923, unable to use his powers?")
    assert (pq.year_from, pq.year_to) == (None, None) and pq.intent == "scene_lookup"
    pq = P(world, "titles with at least 2000 episodes")
    assert (pq.year_from, pq.year_to) == (None, None) and pq.min_episodes == 2000
    pq = P(world, "anime from 2015 or later but 2010 or earlier")
    assert pq.needs_clarification and "Which limit" in pq.clarification


def test_score_phrases(world):
    assert P(world, "Show comedy anime scoring at least 8.").min_score == 8.0      # sentence-final period
    assert P(world, "Find drama movies scoring at least 8.5.").min_score == 8.5
    pq = P(world, "Which TV anime score at least 9?")
    assert pq.types == ["TV"] and pq.min_score == 9.0
    pq = P(world, "Find an anime with an official score of at least 11 out of 10.")
    assert pq.min_score == 11.0 and any("1 to 10" in n for n in pq.notes)
    pq = P(world, "Find movies rated exactly 10 out of 10 in this catalog.")
    assert pq.min_score == pq.max_score == 10.0 and pq.types == ["Movie"]
    pq = P(world, "Find romance titles with no more than 24 episodes.")
    assert pq.max_episodes == 24 and pq.min_episodes is None                     # not "more than 24"
    assert P(world, "no fewer than 5 episodes").min_episodes == 5


def test_studio_cues(world):
    pq = P(world, "action anime produced by Madhouse rated at least 8")
    assert ["studio", "Madhouse"] in pq.include_tags and pq.min_score == 8.0 and not pq.reference_titles
    pq = P(world, "Find anime produced by Nonexistent Evaluation Studio 42.")
    assert pq.unresolved_studios == ["Nonexistent Evaluation Studio 42"]
    assert not any(t[0] == "studio" for t in pq.include_tags)
    assert ["studio", "Bones"] in P(world, "a comedy made by Bones").include_tags
    pq = P(world, "a boy defending his village from a corrupted boar")         # weak cue, no studio
    assert not pq.unresolved_studios and not any(t[0] == "studio" for t in pq.include_tags)
    assert ["studio", "Madhouse"] in P(world, "action, but not by Madhouse").exclude_tags


@pytest.mark.parametrize("text,field", [
    ("Give exact video timecodes for the documented scenes.", "video timecodes"),
    ("List the anime that I personally added to my favorites.", "a favourites list"),
    ("Give the calendar date on which each individual user rating was submitted.", "rating dates"),
    ("What streaming license fee applies to each anime in my country?", "licence fees and prices"),
    ("where can I stream Old Guard?", "streaming availability"),
])
def test_requests_for_data_the_catalogue_does_not_hold(world, text, field):
    pq = P(world, text)
    assert pq.intent == "unavailable_data" and field in pq.unavailable


def test_scene_question_asking_for_the_time_stays_a_scene_lookup(world):
    pq = P(world, "which episode has the scene where they fight on the roof, and what is the timestamp?")
    assert pq.intent == "scene_lookup" and not pq.unavailable


def test_plot_descriptions_and_tag_words_are_not_titles(world):
    pq = P(world, "Where does a boy meet a girl under a bridge and suddenly have a robot emerge from his forehead?")
    assert pq.intent == "scene_lookup" and not pq.reference_titles             # "a girl" is not "A-Girl"
    pq = P(world, "List TV series with a verified School theme.")
    assert ["theme", "School"] in pq.include_tags and not pq.reference_titles  # the tag, not the title "School"
    pq = P(world, "I would like an anime combining action and comedy.")
    assert not pq.needs_clarification and not pq.reference_titles             # "would like" is not "like X"
    assert ["genre", "Action"] in pq.include_tags and ["genre", "Comedy"] in pq.include_tags
    assert P(world, "What anime has the highest score?").intent != "scene_lookup"
    assert P(world, "List Hentai-tagged titles.").include_tags == [["genre", "Hentai"]]


def test_engine_year_and_score_masks(world):
    snap = world[0]
    ce = ConstraintEngine(snap)

    def titles(cs):
        return sorted(snap.titles[i] for i in range(snap.n) if ce.evaluate(cs).mask[i])

    assert titles(ConstraintSet(year_from=2010)) == ["A-Girl", "Mid Point", "New Wave"]   # unknown year fails
    assert titles(ConstraintSet(year_to=2003)) == ["Old Guard", "School"]
    assert titles(ConstraintSet(year_from=2003, year_to=2010)) == ["Mid Point", "School"]
    assert titles(ConstraintSet(min_score=9.0, max_score=9.0)) == ["No Date"]
    cs = ConstraintSet(year_from=2030)
    assert ce.evaluate(cs).count == 0 and "filter.year_from" in cs.active_rules()
    opts = ce.suggest_relaxations(cs, k=3)
    assert opts[0]["rule_id"] == "filter.year_from" and opts[0]["change"] == {"year_from": None}
    assert cs.to_public()["year_from"] == 2030


def test_conversation_keeps_and_clears_year_limits(world):
    st, _, _ = merge(None, P(world, "action anime from 2010 or later"), "action anime from 2010 or later")
    assert st["year_from"] == 2010 and {"kind": "hard", "label": "Began airing 2010 or later"} in constraint_chips(st)
    st, changes, refinement = merge(st, P(world, "any year"), "any year")
    assert refinement and st["year_from"] is None and "no year limit" in changes
    st, _, _ = merge(None, P(world, "comedy anime from 2010"), "comedy anime from 2010")
    assert (st["year_from"], st["year_to"]) == (2010, 2010)
    assert {"kind": "hard", "label": "Began airing 2010"} in constraint_chips(st)
    st, _, _ = merge(st, P(world, "and 2005 or later"), "and 2005 or later")
    assert (st["year_from"], st["year_to"]) == (2005, None)       # an open limit replaces the single year
    st, _, _ = merge(None, P(world, "movies rated exactly 10"), "movies rated exactly 10")
    assert {"kind": "hard", "label": "Score = 10.0"} in constraint_chips(st)


def test_a_setting_period_or_an_age_is_not_an_airing_year(world):
    assert years(world, "a war drama set from 1941 to 1945") == (None, None)
    assert years(world, "a story that takes place in 1923 Tokyo") == (None, None)
    assert years(world, "a comedy about office workers in their 50s") == (None, None)
    assert years(world, "a comedy from the 90s about office workers in their 50s") == (1990, 1999)


def test_prequels_count_as_watch_first_only_in_release_order():
    from app.graph.relations import Edge, PrequelGraph

    # S1 (2020) has sequel S2 (2023); M0 (2021) is a story prequel of S1 but came out after it
    edges = [Edge("s1", "s2", "sequel", "ref"), Edge("s1", "m0", "prequel", "ref"), Edge("s2", "s1", "prequel", "ref")]
    years = {"s1": 2020.0, "s2": 2023.0, "m0": 2021.0}
    g = PrequelGraph(edges, years)
    assert g.chain("s1")[0] == [] and g.later_prequels("s1") == ["m0"]
    assert g.chain("s2")[0] == ["s1"]                          # m0 is not pulled in through s1
    assert PrequelGraph(edges).chain("s1")[0] == ["m0"]        # without years: every prequel counts (old rule)
    assert PrequelGraph(edges, {"s1": 2020.0}).chain("s1")[0] == ["m0"]   # unknown year: kept


def test_quoted_nicknames_in_a_plot_question_are_not_titles(world):
    pq = P(world, 'Which anime has a story where a schoolboy nicknamed "School" hides his psychic powers from everyone?')
    assert pq.intent == "scene_lookup" and not pq.reference_titles
    assert P(world, 'tell me about "Old Guard"').reference_titles[0]["title"] == "Old Guard"   # quotes still work
