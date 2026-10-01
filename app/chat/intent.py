"""Deterministic intent and constraint parser (always available; LLM parsing is optional).

Rules of interpretation
  * "at most / no more than / up to / 最多 / 以内"  -> hard maximum (inclusive)
  * "under / less than / fewer than"               -> hard maximum (exclusive -> N - 1)
  * "no X / without X / don't want X / 不要X"       -> hard exclusion
  * "preferably / ideally / if possible / 最好 / 尽量" before a tag -> soft preference
  * "short / not too long / 不要太长"               -> soft length preference (never silently "12 episodes")
  * "shorter than <reference>"                       -> soft preference relative to the reference
  * a bare genre mention in a request               -> hard include
Titles are always resolved back to catalogue items; nothing is taken from free text on trust.
Chinese input: numbers, tags and reference titles are understood; free-text plot descriptions
in Chinese are NOT matched against the English corpus (reported as a limitation).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.core.text import is_cjk, nfkc, norm

WORD_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
ZH_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(WORD_NUM) + r")"

EN_SYNONYMS: dict[str, list[str]] = {
    "Horror": ["horror", "scary", "creepy", "frightening", "spooky"],
    "Comedy": ["comedy", "comedies", "funny", "hilarious", "comedic", "humor", "humour"],
    "Romance": ["romance", "romantic", "love story", "love stories"],
    "Sci-Fi": ["sci-fi", "sci fi", "scifi", "science fiction"],
    "Slice of Life": ["slice of life", "slice-of-life", "everyday life"],
    "Sports": ["sports", "sport"],
    "Mystery": ["mystery", "mysteries", "whodunit"],
    "Suspense": ["suspense", "thriller", "thrillers"],
    "Fantasy": ["fantasy"],
    "Action": ["action", "fighting", "fights", "battles", "combat"],
    "Adventure": ["adventure", "adventures"],
    "Drama": ["drama", "dramas"],
    "Supernatural": ["supernatural", "ghosts", "spirits"],
    "Psychological": ["psychological", "mind games", "mind-games"],
    "Time Travel": ["time travel", "time-travel", "time loop"],
    "Isekai": ["isekai", "another world", "transported to another world"],
    "Mecha": ["mecha", "giant robots", "robots", "mechs"],
    "School": ["school", "high school", "highschool"],
    "Music": ["music", "musical", "band", "idols"],
    "Military": ["military", "war"],
    "Space": ["space", "outer space"],
    "Gourmet": ["gourmet", "cooking", "food"],
    "Iyashikei": ["iyashikei", "healing", "relaxing", "wholesome"],
    "Martial Arts": ["martial arts"],
    "Historical": ["historical", "history"],
    "Survival": ["survival"],
    "Detective": ["detective", "detectives"],
    "Strategy Game": ["strategy game", "mind game", "strategy"],
    "Super Power": ["super power", "superpowers", "super powers"],
    "Shounen": ["shounen", "shonen"],
    "Seinen": ["seinen"],
    "Shoujo": ["shoujo", "shojo"],
    "Josei": ["josei"],
    "Kids": ["for kids", "for children", "kids"],
    "Award Winning": ["award winning", "award-winning"],
    "Avant Garde": ["avant garde", "avant-garde", "experimental"],
    "Boys Love": ["boys love", "boys' love"],
    "Girls Love": ["girls love", "girls' love", "yuri"],
}
ZH_SYNONYMS: dict[str, list[str]] = {
    "Horror": ["恐怖"], "Comedy": ["喜剧", "搞笑"], "Romance": ["恋爱", "爱情"], "Sci-Fi": ["科幻"],
    "Mystery": ["悬疑", "推理"], "Detective": ["侦探"], "Fantasy": ["奇幻", "魔法"],
    "Action": ["动作", "战斗", "打斗", "热血"], "Adventure": ["冒险"], "Slice of Life": ["日常"],
    "Sports": ["运动", "体育"], "Supernatural": ["超自然"], "Drama": ["剧情"], "Psychological": ["心理"],
    "School": ["校园"], "Mecha": ["机甲"], "Music": ["音乐"], "Isekai": ["异世界"],
    "Time Travel": ["时间旅行", "时间循环", "穿越"], "Iyashikei": ["治愈"], "Gourmet": ["美食"],
    "Military": ["军事", "战争"], "Space": ["宇宙", "太空"], "Historical": ["历史"], "Survival": ["生存"],
    "Suspense": ["惊悚"],
}
NEGATORS = r"(?:no|not|without|avoid|avoiding|except|excluding|exclude|don'?t want|do not want|don'?t like|" \
           r"never|nothing|non|minus|hate|skip|不要|不想要|不想看|不想|别|没有|不含|排除|除了)"
SOFT = r"(?:preferably|ideally|maybe|if possible|a bit of|a little|some|bonus if|prefer(?:ably)?|would like some|" \
       r"最好|尽量|有点|希望有)"
TYPE_WORDS = {"Movie": r"\b(?:movies?|films?)\b|电影|剧场版", "TV": r"\btv (?:series|shows?)\b|电视动画|番剧",
              "OVA": r"\bovas?\b", "ONA": r"\bonas?\b|网络动画", "Special": r"\bspecials?\b"}


@dataclass
class ParsedQuery:
    intent: str = "preference_recommendation"
    reference_titles: list[dict] = field(default_factory=list)   # {item_key, title, match, mention}
    ambiguous_titles: list[dict] = field(default_factory=list)
    include_tags: list[list[str]] = field(default_factory=list)   # [namespace, value]
    exclude_tags: list[list[str]] = field(default_factory=list)
    soft_tags: list[list[str]] = field(default_factory=list)
    soft_avoid_tags: list[list[str]] = field(default_factory=list)
    max_episodes: int | None = None
    min_episodes: int | None = None
    min_score: float | None = None
    types: list[str] | None = None
    budget_minutes: int | None = None
    max_titles: int | None = None
    diversity: str | None = None
    soft_preferences: list[str] = field(default_factory=list)
    hard_constraints: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification: str | None = None
    free_text: str = ""
    language: str = "en"
    reset: bool = False
    parser: str = "deterministic_v1"
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return asdict(self)


class TagLexicon:
    """Maps phrases (EN + limited ZH) to catalogue tags that exist in the active snapshot."""

    def __init__(self, snap):
        self.entries: list[tuple[str, str, str]] = []   # (phrase_norm, namespace, value)
        present: dict[str, tuple[str, str]] = {}
        for ns in ("genre", "theme", "demographic"):
            tm = snap.tags.get(ns)
            if tm is None:
                continue
            for v in tm.vocab:
                present.setdefault(norm(v), (ns, v))
                self.entries.append((norm(v), ns, v))
        for canonical, phrases in list(EN_SYNONYMS.items()) + list(ZH_SYNONYMS.items()):
            hit = present.get(norm(canonical))
            if hit is None:
                continue
            for ph in phrases:
                self.entries.append((norm(ph), hit[0], hit[1]))
        # longest phrases first so "science fiction" wins over "science"
        self.entries.sort(key=lambda e: (-len(e[0]), e[0]))

    def find(self, text: str) -> list[tuple[int, int, str, str]]:
        """Non-overlapping (start, end, namespace, value) spans in normalised text."""
        t = norm(text)
        used = [False] * len(t)
        out = []
        for phrase, ns, val in self.entries:
            if not phrase:
                continue
            pattern = re.escape(phrase) if is_cjk(phrase) else r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])"
            for m in re.finditer(pattern, t):
                if any(used[m.start():m.end()]):
                    continue
                for k in range(m.start(), m.end()):
                    used[k] = True
                out.append((m.start(), m.end(), ns, val))
        return sorted(out)


def _num(s: str) -> float:
    s = s.strip().lower()
    if s in WORD_NUM:
        return float(WORD_NUM[s])
    if s in ZH_NUM:
        return float(ZH_NUM[s])
    return float(s)


def _extract_reference_spans(text: str) -> list[tuple[str, int]]:
    """Candidate reference titles with their start offset in `text`."""
    refs = []
    for m in re.finditer(r"[\"“”「『《]([^\"“”」』》]{2,80})[\"“”」』》]", text):
        refs.append((m.group(1).strip(), m.start(1)))
    pat = (r"(?:\blike|\bsimilar to|\bsuch as|\bsame as|\bin the vein of|\breminds me of|\bcomparable to|"
           r"\bsequel (?:to|of)|\bprequel (?:to|of)|\bseason \d+ of|\btell me about|\binfo (?:on|about)|"
           r"\bwhat do you know about|\bepisodes (?:of|does|in)|\bwatch order (?:of|for)|\bbefore watching|"
           r"\bloved|\bliked|\benjoyed|\bfan of|类似|像|跟|喜欢|看过)\s*"
           r"(.+?)(?=\s+(?:but|and|with|except|without|though|however|that|which|who|has|have|is|are|"
           r"in terms|for me|please)\b|[,.;!?，。；！？]|但|不过|那样|一样|的动漫|的番|$)")
    for m in re.finditer(pat, text, re.IGNORECASE):
        cand = m.group(1).strip(" '\"")
        if 2 <= len(cand) <= 80:
            refs.append((cand, m.start(1)))
    m = re.match(r"^\s*(?:what is|what's|who made|how many episodes (?:does|is|are in)|when did)\s+(.+?)[?？]?\s*$",
                 text, re.IGNORECASE)
    if m:
        refs.append((m.group(1), m.start(1)))
    return refs


def parse(message: str, snap, titles, lexicon: TagLexicon) -> ParsedQuery:
    text = nfkc(message).strip()
    low = text.lower()
    pq = ParsedQuery(language="zh" if is_cjk(text) and not re.search(r"[A-Za-z]{4,}", text) else "en")
    if re.search(r"\b(start over|reset|new search|clear (?:the )?filters)\b|重新开始|清空条件", low):
        pq.reset = True

    # ------------------------------------------------------------ numbers
    for m in re.finditer(rf"(?:at most|no more than|up to|maximum(?: of)?|max\.?|<=)\s*{NUM}\s*(?:episodes?|eps?)\b",
                         low):
        pq.max_episodes = int(_num(m.group(1)))
    for m in re.finditer(rf"(?:under|less than|fewer than|<)\s*{NUM}\s*(?:episodes?|eps?)\b", low):
        pq.max_episodes = int(_num(m.group(1))) - 1
    for m in re.finditer(rf"{NUM}\s*(?:episodes?|eps?)\s*(?:or (?:less|fewer)|max|maximum|at most|or under)", low):
        pq.max_episodes = int(_num(m.group(1)))
    for m in re.finditer(rf"(?:at least|no fewer than|minimum(?: of)?|min\.?|>=)\s*{NUM}\s*(?:episodes?|eps?)\b", low):
        pq.min_episodes = int(_num(m.group(1)))
    for m in re.finditer(rf"(?:more than|over)\s*{NUM}\s*(?:episodes?|eps?)\b", low):
        pq.min_episodes = int(_num(m.group(1))) + 1
    for m in re.finditer(r"(?:最多|不超过|少于)\s*(\d+|[一二两三四五六七八九十])\s*集|(\d+)\s*集(?:以内|以下|之内)", text):
        v = m.group(1) or m.group(2)
        pq.max_episodes = int(_num(v)) - (1 if "少于" in m.group(0) else 0)
    for m in re.finditer(r"(?:至少|不少于)\s*(\d+)\s*集|(\d+)\s*集以上", text):
        pq.min_episodes = int(_num(m.group(1) or m.group(2)))
    for m in re.finditer(r"(?:score|rated|rating|mal)\s*(?:of\s*)?(?:at least|above|over|>=|>|of|≥)?\s*"
                         r"(\d(?:\.\d+)?)\s*(\+|or (?:higher|more|above))?", low):
        v = float(m.group(1))
        if 1 <= v <= 10 and re.search(r"at least|above|over|>=|>|\+|or higher|or more|or above|≥", m.group(0)):
            pq.min_score = v
    for m in re.finditer(r"(\d(?:\.\d+)?)\s*(?:\+|or (?:higher|above))\s*(?:score|rating|rated)", low):
        pq.min_score = float(m.group(1))
    for m in re.finditer(r"评分\s*(?:在|不低于|大于|高于|超过)?\s*(\d(?:\.\d+)?)\s*(?:分)?\s*(?:以上|及以上)?|"
                         r"(\d(?:\.\d+)?)\s*分以上", text):
        pq.min_score = float(m.group(1) or m.group(2))
    hours = re.search(rf"{NUM}\s*(?:hours?|hrs?|h)\b(?:\s*(?:and\s*)?(\d+)\s*(?:minutes?|mins?))?", low)
    mins = re.search(r"\b(\d+)\s*(?:minutes?|mins?)\b", low)
    zh_hours = re.search(r"(\d+(?:\.\d+)?|[一二两三四五六七八九十])\s*(?:个)?小时", text)
    if hours:
        pq.budget_minutes = int(round(_num(hours.group(1)) * 60 + (int(hours.group(2)) if hours.group(2) else 0)))
    elif zh_hours:
        pq.budget_minutes = int(round(_num(zh_hours.group(1)) * 60))
    elif mins and re.search(r"\b(?:plan|budget|have|free|spare|time)\b", low):
        pq.budget_minutes = int(mins.group(1))
    m = re.search(rf"(?:at most|no more than|up to|max(?:imum)?(?: of)?)\s*{NUM}\s*(?:titles?|shows?|series|anime|works)",
                  low) or re.search(r"最多\s*(\d+|[一二两三四五六七八九十])\s*部", text)
    if m:
        pq.max_titles = int(_num(m.group(1)))
    if re.search(r"different styles?|varied|variety|diverse|don'?t repeat|no repeats?|not the same|mix it up|"
                 r"风格(?:尽量)?不同|不要重复|多样", low):
        pq.diversity = "high"

    # ------------------------------------------------------------ types
    for t, pat in TYPE_WORDS.items():
        if re.search(pat, low) and not re.search(rf"{NEGATORS}\s+(?:a\s+)?(?:{pat})", low):
            if t == "Movie" and re.search(r"like the (?:movie|film)", low):
                continue
            pq.types = sorted(set((pq.types or []) + [t]))

    # ------------------------------------------------------------ reference titles
    seen = set()
    for raw, start in _extract_reference_spans(text):
        # prefer the longest exact alias starting at the same place: the clause splitter cuts at "!" or ","
        exact_span = titles.longest_prefix_mention(text[start:])
        if exact_span and len(exact_span) >= len(raw):
            raw = exact_span
        matches = titles.resolve(raw, limit=3, fuzzy_cutoff=85.0)
        if not exact_span:
            # a typo inside a title with punctuation ("Komi-san wa, Cmyushou desu."): also try the span up to the
            # next clause word, and keep whichever resolves with the higher score
            clause = re.match(r"(.+?)(?=\s+(?:but|and|with|except|without|though|however|that|which|who|please)\b"
                              r"|[?？。]|$)", text[start:], re.IGNORECASE)
            longer = clause.group(1).strip(" '\"") if clause else ""
            if len(longer) > len(raw) and len(longer) <= 80:
                alt = titles.resolve(longer, limit=3, fuzzy_cutoff=85.0)
                if alt and (not matches or alt[0].score > matches[0].score):
                    raw, matches = longer, alt
        if not matches:
            continue
        top = matches[0]
        if top.item_key in seen:
            continue
        if top.match == "fuzzy" and len(matches) > 1 and matches[1].score >= top.score - 2 \
                and matches[1].item_key != top.item_key:
            pq.ambiguous_titles.append({"mention": raw, "options": [
                {"item_key": m.item_key, "title": m.title} for m in matches[:3]]})
            continue
        seen.add(top.item_key)
        pq.reference_titles.append({"item_key": top.item_key, "title": top.title, "match": top.match,
                                    "mention": raw})
    if not pq.reference_titles:
        for m, _span in titles.find_mentions(text):
            if m.item_key not in seen:
                seen.add(m.item_key)
                pq.reference_titles.append({"item_key": m.item_key, "title": m.title, "match": m.match,
                                            "mention": m.alias})
    ref_spans = [r["mention"].lower() for r in pq.reference_titles]

    # ------------------------------------------------------------ tags (negation / soft markers)
    t_norm = norm(text)
    for start, end, ns, val in lexicon.find(text):
        span_text = t_norm[start:end]
        if any(span_text in r for r in ref_spans):
            continue    # part of a title mention, not a tag request
        before = t_norm[max(0, start - 28):start]
        pair = [ns, val]
        if re.search(r"not too\s*$|不要太\s*$", before):
            pq.soft_avoid_tags.append(pair)
        elif re.search(rf"{NEGATORS}\b[\w\s'-]{{0,12}}$|{NEGATORS}\s*$", before):
            pq.exclude_tags.append(pair)
        elif re.search(rf"{SOFT}\b[\w\s'-]{{0,16}}$|{SOFT}\s*$", before):
            pq.soft_tags.append(pair)
        else:
            pq.include_tags.append(pair)

    # ------------------------------------------------------------ soft length / quality preferences
    if re.search(r"\bshorter\b|更短|短一点|短一些", low) and pq.reference_titles:
        pq.soft_preferences.append("shorter_than_reference")
    elif re.search(r"\b(?:short|quick|brief|not too long|not very long|bite-sized)\b|不要太长|短篇|短的", low):
        pq.soft_preferences.append("short")
    if re.search(r"\blonger\b|更长", low) and pq.reference_titles:
        pq.soft_preferences.append("longer_than_reference")
    elif re.search(r"\b(?:long|lengthy|epic)\b(?! story)|长篇", low) and "short" not in pq.soft_preferences:
        pq.soft_preferences.append("long")
    if re.search(r"\b(?:hidden gems?|underrated|obscure|lesser[- ]known)\b|冷门|小众", low):
        pq.soft_preferences.append("obscure")
    if re.search(r"\b(?:popular|famous|well[- ]known|classic)\b|热门|经典", low):
        pq.soft_preferences.append("popular")
    if re.search(r"\b(?:highly rated|best|top rated|top-rated|critically acclaimed)\b|高分|神作", low) and \
            pq.min_score is None:
        pq.soft_preferences.append("high_score")

    # ------------------------------------------------------------ intent
    has_ref = bool(pq.reference_titles)
    if re.search(r"screenshot|this (?:image|picture|frame|screen ?cap)|attached image|截图|这张图|图片里", low):
        pq.intent = "screenshot_lookup"
    elif pq.budget_minutes and re.search(r"plan|schedule|weekend|marathon|binge|hours|watch list|watchlist|"
                                         r"计划|周末|安排|小时", low):
        pq.intent = "viewing_plan"
    elif re.search(r"which episode|what episode|the scene|scene where|scene in which|哪一集|第几集|名场面|场景", low):
        pq.intent = "scene_lookup"
    elif has_ref and re.search(r"sequel|prequel|watch order|what order|order to watch|before watching|"
                               r"season \d|followed by|续作|前传|观看顺序|先看", low):
        pq.intent = "relation_query"
    elif has_ref and re.search(r"\blike\b|similar|such as|same as|reminds me|in the vein|comparable|loved|liked|"
                               r"enjoyed|fan of|类似|像|差不多|喜欢", low):
        pq.intent = "similar_to"
    elif has_ref and (re.search(r"tell me about|what is|what's|info|how many episodes|who made|which studio|"
                                r"score of|when did|when was|is it good|介绍|多少集|评分|哪个工作室", low)
                      or len(text.split()) <= 6):
        pq.intent = "anime_fact"
    elif _out_of_scope(low) and not (pq.include_tags or pq.exclude_tags or pq.max_episodes or has_ref):
        pq.intent = "out_of_scope"
    else:
        pq.intent = "preference_recommendation"

    if pq.min_episodes is not None and pq.max_episodes is not None and pq.min_episodes > pq.max_episodes:
        pq.needs_clarification = True
        pq.clarification = (f"You asked for at least {pq.min_episodes} but at most {pq.max_episodes} episodes. "
                            "Which limit should I keep?")
    overlap = {tuple(t) for t in pq.include_tags} & {tuple(t) for t in pq.exclude_tags}
    if overlap:
        pq.needs_clarification = True
        pq.clarification = f"You both asked for and excluded {', '.join(v for _, v in overlap)}. Which one do you mean?"
    if pq.ambiguous_titles and not pq.reference_titles:
        pq.needs_clarification = True
        opts = "; ".join(o["title"] for o in pq.ambiguous_titles[0]["options"])
        pq.clarification = f'"{pq.ambiguous_titles[0]["mention"]}" could be several titles: {opts}. Which one?'

    for name, v in (("max_episodes", pq.max_episodes), ("min_episodes", pq.min_episodes),
                    ("min_score", pq.min_score), ("types", pq.types), ("include_tags", pq.include_tags),
                    ("exclude_tags", pq.exclude_tags), ("budget_minutes", pq.budget_minutes),
                    ("max_titles", pq.max_titles)):
        if v not in (None, []):
            pq.hard_constraints.append(name)
    if has_ref and pq.intent == "similar_to":
        pq.soft_preferences.insert(0, "similar_to_reference")
    pq.free_text = _free_text(text, pq)
    if pq.language == "zh" and len(pq.free_text) > 6:
        pq.notes.append("Chinese plot descriptions are not matched against the English knowledge base; "
                        "numbers, tags and titles were understood.")
    return pq


def _out_of_scope(low: str) -> bool:
    return bool(re.search(r"\b(?:weather|stock|bitcoin|crypto|recipe for|write (?:me )?(?:code|an essay|a poem)|"
                          r"translate|homework|math problem|news today|capital of|president)\b", low))


def _free_text(text: str, pq: ParsedQuery) -> str:
    """Query text for retrieval with constraint phrases removed (titles kept out of the dense query)."""
    t = text
    for r in pq.reference_titles:
        t = re.sub(re.escape(r["mention"]), " ", t, flags=re.IGNORECASE)
    t = re.sub(rf"(?:at most|no more than|up to|under|less than|fewer than|at least|more than|over)\s*{NUM}\s*"
               r"(?:episodes?|eps?)", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:score|rated|rating)\s*(?:of\s*)?(?:at least|above|over)?\s*\d(?:\.\d+)?\+?", " ", t,
               flags=re.IGNORECASE)
    t = re.sub(rf"{NUM}\s*(?:hours?|hrs?|minutes?|mins?)", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:i want|i'd like|looking for|recommend(?: me)?|give me|find me|suggest|can you|please|"
               r"something|anime|show|series)\b", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip(" ,.;!?")
