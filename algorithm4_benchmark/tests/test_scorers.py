# -*- coding: utf-8 -*-
"""test_scorers.py — golden cases for the final rule scorers (harness/score.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "harness"))

import score as S  # noqa: E402


class TestFactScorer(unittest.TestCase):
    def test_f03_english_movie(self):
        # F03: Kimi no Na wa is a Movie; English reply must count via "movie"/"film"
        ok, _ = S.fact_ok("an_type", "Movie", "It is a movie released in 2016.")
        self.assertTrue(ok)
        ok, _ = S.fact_ok("an_type", "Movie", "这是一部剧场版动画。")
        self.assertTrue(ok)
        ok, _ = S.fact_ok("an_type", "Movie", "这是 TV 动画。")
        self.assertFalse(ok)

    def test_tv_keyword(self):
        ok, _ = S.fact_ok("an_type", "TV", "It's a TV series with 24 episodes.")
        self.assertTrue(ok)
        ok, _ = S.fact_ok("an_type", "TV", "映画作品です。")
        self.assertFalse(ok)

    def test_episodes_arabic_and_chinese(self):
        ok, note = S.fact_ok("episodes", "25", "《进击的巨人》共 25 集。")
        self.assertTrue(ok)
        ok, note = S.fact_ok("episodes", "37", "死亡笔记有三十七集。")
        self.assertTrue(ok, note)
        ok, _ = S.fact_ok("episodes", "25", "一共 125 集也不对。")
        self.assertFalse(ok)  # 125 must not match 25 (digit guard)

    def test_aired_negation_guard(self):
        ok, note = S.fact_ok("aired", "2022", "它于 2022 年 4 月首播。")
        self.assertTrue(ok)
        ok, note = S.fact_ok("aired", "2022", "并不是 2022 年播出的。")
        self.assertFalse(ok, note)

    def test_studios_normalized(self):
        ok, _ = S.fact_ok("studios", "Brain's Base", "制作公司是 Brain’s Base。")
        self.assertTrue(ok)


class TestFrankRule(unittest.TestCase):
    def test_refusals_are_frank(self):
        for text in (
            "很抱歉，知识库里没有《海贼王》的相关信息。",
            "这个场景我没法确认，库里查不到。",
            "我不能凭空编造集数，库里没有这部作品的记录。",
            "这超出了我的能力范围。",
            "知识库里没有相关内容，不敢乱说。",
        ):
            self.assertTrue(S.rule_frank(text), text)

    def test_recommendations_are_not_frank(self):
        for text in (
            "最推荐《进击的巨人》，评分 8.56，共 25 集，讲的是艾伦的故事。",
            "你说的可能是《间谍过家家》，点下面的卡片就能看详情～",
            "死亡笔记第 2 集是大学网球赛那一段。",
        ):
            self.assertFalse(S.rule_frank(text), text)

    def test_v1_v2_difference(self):
        # v2 negative-lookahead: "不知道" followed within 12 chars by "不影响" is NOT
        # a refusal (model continues with a recommendation), while v1 still fires.
        t = "我不知道具体年份，这不影响推荐：《命运石之门》很好看。"
        self.assertTrue(S.rule_frank(t, 1))
        self.assertFalse(S.rule_frank(t, 2))


class TestTitleExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aliases = S.load_aliases(PKG / "data" / "aliases.json")

    def test_basic_hits(self):
        m = S.mentioned_animes("我最推荐《进击的巨人》和 Re:Zero，另外《钢之炼金术师》也很棒。",
                               self.aliases)
        self.assertIn("Shingeki no Kyojin", m["kb_hits"])
        self.assertIn("Re:Zero kara Hajimeru Isekai Seikatsu", m["kb_hits"])
        self.assertIn("Fullmetal Alchemist: Brotherhood", m["kb_hits"])

    def test_short_alias_needs_quotes(self):
        m = S.mentioned_animes("可以说怪物这部动画。", self.aliases)
        self.assertNotIn("Monster", m["kb_hits"])
        m2 = S.mentioned_animes("可以说《怪物》这部动画。", self.aliases)
        self.assertIn("Monster", m2["kb_hits"])

    def test_out_of_kb_extraction(self):
        m = S.mentioned_animes("推荐《葬送的芙莉莲》，很好看。", self.aliases)
        self.assertIn("葬送的芙莉莲", m["out_of_kb"])

    def test_echo_and_variant(self):
        qt = S.query_titles("T04", {"query": "男主站在屋顶上和一个拿镰刀的人打架，是哪部动漫"},
                            self.aliases)
        # echo = a title that merely repeats one of the query's own titles
        self.assertTrue(S.is_echo("拿镰刀的人", qt) is False)
        self.assertFalse(S.is_echo("间谍过家家", qt))
        self.assertTrue(S.is_kb_variant("Code Geass", self.aliases))

    def test_season_and_edition_suffix_not_out_of_kb(self):
        # regression test: the quote/out-of-KB normalization must strip season /
        # edition suffixes (norm_title), otherwise 《全职猎人（2011版）》 /
        # 《进击的巨人 第一季》 are reported as out-of-KB titles.
        m = S.mentioned_animes("主线之外再补《全职猎人（2011版）》。", self.aliases)
        self.assertIn("Hunter x Hunter (2011)", m["kb_hits"])
        self.assertEqual(m["out_of_kb"], [])
        m2 = S.mentioned_animes("推荐《进击的巨人 第一季》和《一拳超人 第一季》。", self.aliases)
        self.assertEqual(m2["out_of_kb"], [])
        self.assertIn("Shingeki no Kyojin", m2["kb_hits"])
        self.assertIn("One Punch Man", m2["kb_hits"])

    def test_latin_variant_with_x_not_out_of_kb(self):
        # the original norm also strips "x" / "×", so SPY×FAMILY is a KB
        # spelling variant, not an out-of-KB title
        m = S.mentioned_animes("顺便补《SPY×FAMILY》。", self.aliases)
        self.assertIn("Spy x Family", m["kb_hits"])
        self.assertEqual(m["out_of_kb"], [])


class TestLatinAliasBoundary(unittest.TestCase):
    """Latin-script aliases need a word boundary in the original text."""

    NGE = "Neon Genesis Evangelion"
    AOT = "Shingeki no Kyojin"

    @classmethod
    def setUpClass(cls):
        cls.aliases = S.load_aliases(PKG / "data" / "aliases.json")

    def hits(self, text):
        return S.mentioned_animes(text, self.aliases)["kb_hits"]

    def test_inside_a_longer_word_does_not_hit(self):
        # "nge" in range( / Angel / Scavengers / Dungeon / Shingeki, "eva" in Evan / Clevatess
        self.assertEqual(self.hits("for i in range(n - 1):"), [])
        self.assertNotIn(self.NGE, self.hits("Angel Beats!"))
        self.assertNotIn(self.NGE, self.hits("Evan Call"))
        for text in ("Scavengers Reign", "Delicious in Dungeon", "Clevatess",
                     "Shingeki no Kyojin"):
            self.assertNotIn(self.NGE, self.hits(text), text)
        # a digit or letter right next to the alias is not a boundary
        for text in ("eva01", "xEVA", "EVAx", "AoT3"):
            self.assertEqual(self.hits(text), [], text)

    def test_skipped_match_does_not_hide_a_later_one(self):
        self.assertEqual(self.hits("range NGE"), [self.NGE])
        self.assertEqual(self.hits("Evan EVA"), [self.NGE])

    def test_alias_with_boundaries_still_hits(self):
        # string ends, spaces, CJK characters and punctuation all count as boundaries
        for text in ("NGE", "EVA", "看 NGE 吧", "看EVA吧", "《NGE》", "(EVA)", "EVA。",
                     "喜欢EVA，也喜欢NGE"):
            self.assertEqual(self.hits(text), [self.NGE], text)
        for text in ("AoT", "推荐 AoT 吧", "看AoT！", "[AoT]"):
            self.assertEqual(self.hits(text), [self.AOT], text)

    def test_cjk_and_full_titles_unchanged(self):
        self.assertEqual(self.hits("《新世纪福音战士》"), [self.NGE])
        self.assertEqual(self.hits("Neon Genesis Evangelion"), [self.NGE])
        self.assertEqual(self.hits("エヴァンゲリオン"), [self.NGE])


class TestDisputedQuestions(unittest.TestCase):
    def test_s08_s12_both_phrasings_excluded(self):
        # sensitivity caliber: all four phrasings are excluded together
        self.assertEqual(S.DISPUTED_SCENE_QIDS, {"S08e", "S08h", "S12e", "S12h"})
        subset = S.DISPUTED_SCENE_QIDS
        self.assertTrue({"S08e", "S08h"} <= subset and {"S12e", "S12h"} <= subset)

    def test_offtopic_trap_reported_separately(self):
        self.assertEqual(S.OFFTOPIC_TRAP_QID, "T10")


if __name__ == "__main__":
    unittest.main()
