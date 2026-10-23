"""Deterministic intent and constraint parser (always available; LLM parsing is optional).

Rules of interpretation
  * "at most / no more than / up to / 最多 / 以内"  -> hard maximum (inclusive)
  * "under / less than / fewer than"               -> hard maximum (exclusive -> N - 1)
  * "no X / without X / don't want X / 不要X"       -> hard exclusion
  * "preferably / ideally / if possible / 最好 / 尽量" before a tag -> soft preference
  * "short / not too long / 不要太长"               -> soft length preference (never silently "12 episodes")
  * "shorter than <reference>"                       -> soft preference relative to the reference
  * a bare genre mention in a request               -> hard include
  * "in 2010 or later / since 2010 / 2010年以后"        -> first-aired year >= 2010 (inclusive)
  * "2000 or earlier / until 2000"                    -> first-aired year <= 2000; "before 2000" -> <= 1999
  * "produced by / animated by / studio X"           -> studio filter if X is a catalogue studio, otherwise
                                                        reported as an unknown studio (never guessed)
  * questions about data the catalogue does not hold (video timecodes, rating dates, licence fees, streaming
    availability, a "favourites" list) are answered as such instead of being turned into recommendations
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
YEAR = r"((?:19|20)\d{2})"
# a four-digit number followed by a unit is a count, not a year ("at least 2000 episodes")
_NOT_COUNT = r"(?!\s*(?:episodes?|eps?|minutes?|mins?|hours?|hrs?|titles?|shows?|members?|users?|votes?|ratings?|集|部))"
SCORE_VAL = r"(?<![\d.])(\d{1,2}(?:\.\d+)?)(?!\d|\.\d)"   # "at least 8." ends a sentence
SCORE_CUE = r"(?:scor(?:e|es|ed|ing)|rated|rating|mal)"
STRONG_STUDIO_CUE = r"(?:produced|animated) by(?: (?:the )?studio)?|(?:by|from|of) (?:the )?studio|studio:"
WEAK_STUDIO_CUE = r"(?:made|done) by|\bby|\bfrom"
STUDIO_STOP = r"(?=\s+(?:and|with|that|which|but|in|from|after|before|since|or|scoring|rated|having|where|who)\b|" \
              r"\s+(?:at most|at least|no more than|up to)\b|[,.;!?，。；！？]|$)"
# requests for fields that are not in the catalogue -> (field label, pattern)
UNAVAILABLE = (
    ("video timecodes", r"\btime ?codes?\b|\btimestamps?\b|\bexact (?:video )?(?:times?|minutes?|seconds?)\b|"
                        r"\bat what (?:minute|second|time) (?:in|of) the episode\b|时间码|时间戳|几分几秒"),
    ("rating dates", r"\b(?:dates?|days?|times?|when)\b[^.?!]{0,60}\b(?:ratings?|scores?|votes?|reviews?)\b[^.?!]{0,40}"
                     r"\b(?:submitted|given|posted|made|entered|added)\b|评分(?:的)?(?:时间|日期)"),
    ("licence fees and prices", r"\blicen[cs](?:e|ing) (?:fees?|costs?|prices?)\b|\broyalt(?:y|ies)\b|"
                                r"\bsubscription (?:fees?|prices?|costs?)\b|\bhow much (?:does|do|would|will) "
                                r"(?:it|they|\w+) cost\b|版权费|授权费|多少钱"),
    ("streaming availability", r"\bwhere (?:can|could|do|should) i (?:stream|watch|legally watch)\b|"
                               r"\bwhich (?:streaming )?(?:platforms?|services?|sites?|apps?) (?:has|have|streams?|"
                               r"carr(?:y|ies)|offers?)\b|\bstreaming (?:services?|platforms?|rights)\b|"
                               r"在哪(?:里|儿)?(?:看|播)|哪个平台"),
    ("box office and sales", r"\bbox office\b|\bsales (?:figures?|numbers?)\b|\bhow many (?:copies|blu-?rays?|dvds?)\b|"
                             r"\brevenue\b|票房|销量"),
    ("a favourites list", r"\bmy (?:favou?rites?|favs?)\b|\bi (?:personally )?(?:favou?rited|bookmarked)\b|"
                          r"我的收藏|我收藏的"),
)
TYPE_WORDS = {"Movie": r"\b(?:movies?|films?)\b|电影|剧场版", "TV": r"\btv (?:series|shows?|anime|titles?)\b|电视动画|番剧",
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
    max_score: float | None = None
    year_from: int | None = None                  # first-aired year, inclusive
    year_to: int | None = None
    clear_years: bool = False                     # "any year": drop earlier year limits
    types: list[str] | None = None
    unresolved_studios: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)   # requested fields the data does not hold
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
        # studios are matched only after an explicit cue ("produced by", "studio X"): many studio names are
        # ordinary words ("Bones", "Sunrise")
        self.studios: dict[str, str] = {}
        tm = snap.tags.get("studio")
        for v in (tm.vocab if tm is not None else []):
            self.studios.setdefault(norm(v), v)
            bare = re.sub(r"^studios?\s+|\s+studios?$", "", norm(v))
            if bare and bare != norm(v):
                self.studios.setdefault(bare, v)

    def is_tag_phrase(self, text: str) -> bool:
        if not hasattr(self, "_phrases"):
            self._phrases = {e[0] for e in self.entries}
        return norm(text) in self._phrases

    def studio_prefix(self, rest: str) -> tuple[str, int] | None:
        """Longest catalogue studio that `rest` starts with: (value, length in normalised text)."""
        r = norm(rest)
        r = re.sub(r"^(?:the\s+)?studios?\s+", "", r)
        best = None
        for key, v in self.studios.items():
            if len(key) >= 2 and r.startswith(key) and (len(r) == len(key) or not r[len(key)].isalnum()):
                if best is None or len(key) > best[1]:
                    best = (v, len(key))
        return best

    def studio(self, mention: str) -> str | None:
        """Catalogue studio for a mention (exact after normalisation, then a close typo match), else None."""
        key = re.sub(r"^(?:the\s+)?(?:studios?\s+)?|\s+studios?$", "", norm(mention)).strip(" '\"")
        if not key:
            return None
        if key in self.studios:
            return self.studios[key]
        if len(key) < 5:
            return None
        from rapidfuzz import fuzz, process

        hit = process.extractOne(key, list(self.studios), scorer=fuzz.ratio, score_cutoff=90)
        return self.studios[hit[0]] if hit else None

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


def _extract_reference_spans(text: str, quotes: bool = True) -> list[tuple[str, int]]:
    """Candidate reference titles with their start offset in `text`. With `quotes=False` quoted words are not
    candidates (in a plot description they are nicknames or terms: Shigeo "Mob" Kageyama)."""
    refs = []
    for m in re.finditer(r"[\"“”「『《]([^\"“”」』》]{2,80})[\"“”」』》]", text) if quotes else ():
        refs.append((m.group(1).strip(), m.start(1)))
    # "like X" is a reference, "I would like / I'd like / like to" is not
    pat = (r"(?:(?<!would )(?<!'d )(?<!’d )\blike(?! to\b)|\bsimilar to|\bsuch as|\bsame as|\bin the vein of|"
           r"\breminds me of|\bcomparable to|"
           r"\bsequel (?:to|of)|\bprequel (?:to|of)|\bseason \d+ of|\btell me about|\binfo (?:on|about)|"
           r"\bwhat do you know about|\bepisodes? (?:of|does|in)|\bwatch order (?:of|for)|\bbefore watching|"
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


def _parse_years(low: str, text: str, pq: ParsedQuery) -> None:
    """First-aired year limits (inclusive). A plain year in a plot description ("Earth in 1923") is not a limit:
    an open limit needs a cue (since / or later / before ...), an exact year needs an airing verb or a noun.
    When the story is set in a period ("set from 1941 to 1945") or an age is meant ("in their 50s"), the phrase
    describes the content, not the airing date, and is left out."""
    y = YEAR + _NOT_COUNT
    blank = re.compile(r"\b(?:set|takes? place|taking place|happens?|happening|is set)\s+(?:in|during|between|from|"
                       r"around|before|after)\s+[^,.;!?]*|\b(?:their|his|her|my|your|our|early|late|mid)[\s-]+"
                       r"(?:\d0s|twenties|thirties|forties|fifties|sixties)\b|故事(?:发生)?在[^，。,.]*")
    low = blank.sub(lambda m: " " * len(m.group(0)), low)
    text = blank.sub(lambda m: " " * len(m.group(0)), text)
    if re.search(r"\b(?:any|whatever|regardless of(?: the)?) (?:release |airing |start )?years?\b|不限年份|年份不限", low):
        pq.clear_years = True
        return
    m = (re.search(rf"\b(?:between|from)\s+{y}\s*(?:and|to|-|–|until|through)\s*{y}", low)
         or re.search(rf"(?<!\d){y}\s*(?:-|–|to)\s*{y}", low)
         or re.search(rf"{YEAR}\s*年?\s*(?:到|至|-|–)\s*{YEAR}\s*年", text))
    if m:
        pq.year_from, pq.year_to = sorted((int(m.group(1)), int(m.group(2))))
        return
    m = re.search(r"\b(?:the\s+)?((?:19|20)\d)0s\b", low) or re.search(r"(?:^|[\s'‘’])([5-9])0s\b", low)
    zh = re.search(r"((?:19|20)\d|[5-9])0\s*年代", text)
    if m or zh:
        d = (m or zh).group(1)
        start = int(d) * 10 if len(d) == 3 else 1900 + int(d) * 10
        pq.year_from, pq.year_to = start, start + 9
        return
    for m in re.finditer(rf"{y}\s*(?:or|and)\s*(?:later|after|newer|more recent|beyond|onwards?|up)\b|{y}\s*(?:onwards?\b|\+)"
                         rf"|\b(?:since|starting(?: in| from)?|beginning(?: in| from)?|no earlier than|not before|"
                         rf"in or after)\s+(?:the year\s+)?{y}", low):
        pq.year_from = int(next(g for g in m.groups() if g))
    for m in re.finditer(rf"\b(?:after|newer than|later than|post-?)\s*{y}", low):
        pq.year_from = int(m.group(1)) + 1
    for m in re.finditer(rf"{y}\s*(?:or|and)\s*(?:earlier|before|older|prior)\b|\b(?:until|till|up to|through|"
                         rf"no later than|not after|in or before)\s+(?:the year\s+)?{y}", low):
        pq.year_to = int(next(g for g in m.groups() if g))
    for m in re.finditer(rf"\b(?:before|older than|earlier than|prior to|pre-?)\s*{y}", low):
        pq.year_to = int(m.group(1)) - 1
    for m in re.finditer(rf"{YEAR}\s*年?\s*(?:及|或)?(?:以后|之后|以来|起|后)", text):
        pq.year_from = int(m.group(1))
    for m in re.finditer(rf"{YEAR}\s*年?\s*(?:(及|或)(?:以前|之前|更早)|(?:以前|之前|前))", text):
        pq.year_to = int(m.group(1)) - (0 if m.group(2) else 1)
    if pq.year_from is None and pq.year_to is None:
        m = (re.search(rf"\b(?:aired|airing|released|premiered|came out|debuted|broadcast|made|produced|started|began)"
                       rf"\s+(?:in|during)\s+{y}", low)
             or re.search(rf"\b(?:anime|shows?|titles?|series|movies?|films?)\s+(?:from|of)\s+{y}", low)
             or re.search(rf"{YEAR}\s*年(?:的|播出|开播|首播|上映)", text))
        if m:
            pq.year_from = pq.year_to = int(m.group(1))


def _parse_studios(text: str, pq: ParsedQuery, lexicon: TagLexicon) -> list[tuple[int, int]]:
    """Studio filters after an explicit cue. An unknown studio after a strong cue is reported, never guessed.
    Returns the character spans of the studio mentions (they are not title mentions)."""
    seen = {tuple(t) for t in pq.include_tags + pq.exclude_tags}
    spans: list[tuple[int, int]] = []
    for strong, cue in ((True, STRONG_STUDIO_CUE), (False, WEAK_STUDIO_CUE)):
        for m in re.finditer(rf"(?:{cue})\s+", text, re.IGNORECASE):
            rest = text[m.end():]
            hit = lexicon.studio_prefix(rest)
            value = hit[0] if hit else None
            if value is not None and not strong and len(value) < 6 and not rest.startswith(value):
                value = None    # "from a corrupted boar" is not the studio "Arch"; short names need exact case
            mention = None
            if value is None:
                mm = re.match(rf"(.+?){STUDIO_STOP}", rest, re.IGNORECASE)
                mention = mm.group(1).strip(" '\"") if mm else ""
                value = lexicon.studio(mention) if 0 < len(mention) <= 60 else None
            negated = re.search(rf"{NEGATORS}\s*(?:\w+\s+){{0,2}}$", text[:m.start()].lower())
            if value is not None:
                pair = ("studio", value)
                spans.append((m.end(), m.end() + (len(value) if hit else len(mention or ""))))
                if pair not in seen:
                    seen.add(pair)
                    (pq.exclude_tags if negated else pq.include_tags).append(list(pair))
            elif strong and mention and mention not in pq.unresolved_studios:
                pq.unresolved_studios.append(mention)
                spans.append((m.end(), m.end() + len(mention)))
    return spans


def _unavailable(low: str) -> list[str]:
    return [label for label, pat in UNAVAILABLE if re.search(pat, low)]


IDENT_NOUN = r"(?:anime|show|series|film|movie|special|ova|ona|sequel|prequel|adaptation|title|one|story|episode)"
_MODS = r"(?:[\w'-]+\s+){0,3}"


_SCAFFOLD = (
    rf"^\s*(?:which|what)\s+{_MODS}(?:{IDENT_NOUN}\s+)?(?:has|have|had|shows?|showed|features?|opens? with|"
    r"begins? with|starts? with|ends? with|follows?|depicts?|includes?|is about)\s+",
    r"^\s*where\s+(?:do|does|did|is|are|was|were)\s+",
    rf"^\s*in\s+(?:which|what)\s+{_MODS}{IDENT_NOUN}\s+(?:do|does|did|is|are|was|were)\s+",
    rf"^\s*(?:i'?m\s+)?(?:trying to (?:identify|find|remember|recall)|looking for|find|name|identify)\s+"
    rf"(?:the|that|an?|this)\s+{_MODS}{IDENT_NOUN}\s+(?:where|in which|that|with)\s+",
)


def plot_query(text: str) -> str:
    """The plot description of a "which anime has <plot>?" question, without the question scaffold
    (retrieval matches the description against scene records and synopses)."""
    q = text.replace("’", "'").strip()
    for pat in _SCAFFOLD:
        m = re.match(pat, q, re.IGNORECASE)
        if m and len(q[m.end():].split()) >= 4:
            q = q[m.end():]
            break
    return q.strip(" ?.!")


def _identification_question(low: str) -> bool:
    """A long "which anime has <plot / scene description>?" question (find the title, not recommend one)."""
    q = low.replace("’", "'")
    words = len(q.split())
    patterns = (
        (8, r"\b(?:trying to (?:identify|find|remember|recall)|can'?t (?:remember|recall) (?:the )?(?:name|title)|"
            r"what(?:'s| is| was) the (?:name|title) of (?:the|that|this|an?)\b|identify (?:the|an?|this|that) "
            rf"{_MODS}{IDENT_NOUN})\b"),
        (8, rf"^\s*(?:find|name|identify|i'?m looking for|looking for) (?:the|that|an?) {_MODS}{IDENT_NOUN}\s+"
            r"(?:where|in which|that (?:has|shows|opens|begins|starts))\b"),
        (8, rf"^\s*in (?:which|what) {_MODS}{IDENT_NOUN}\b"),
        (10, r"^\s*where (?:do|does|did|is|are|was|were)\b"),
        (10, rf"^\s*(?:which|what) {_MODS}(?:{IDENT_NOUN}\s+)?(?:has|have|had|shows?|showed|features?|opens? with|"
             r"begins? with|starts? with|ends? with|follows?|depicts?|is about)\b"),
    )
    return any(words >= n and re.search(p, q) for n, p in patterns)


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
    for m in re.finditer(rf"(?:under|(?<!no )(?<!not )(?:less|fewer) than|<)\s*{NUM}\s*(?:episodes?|eps?)\b", low):
        pq.max_episodes = int(_num(m.group(1))) - 1
    for m in re.finditer(rf"{NUM}\s*(?:episodes?|eps?)\s*(?:or (?:less|fewer)|max|maximum|at most|or under)", low):
        pq.max_episodes = int(_num(m.group(1)))
    for m in re.finditer(rf"(?:at least|no (?:fewer|less) than|minimum(?: of)?|min\.?|>=)\s*{NUM}\s*(?:episodes?|eps?)\b",
                         low):
        pq.min_episodes = int(_num(m.group(1)))
    for m in re.finditer(rf"(?:(?<!no )(?<!not )more than|over)\s*{NUM}\s*(?:episodes?|eps?)\b", low):
        pq.min_episodes = int(_num(m.group(1))) + 1
    for m in re.finditer(r"(?:最多|不超过|少于)\s*(\d+|[一二两三四五六七八九十])\s*集|(\d+)\s*集(?:以内|以下|之内)", text):
        v = m.group(1) or m.group(2)
        pq.max_episodes = int(_num(v)) - (1 if "少于" in m.group(0) else 0)
    for m in re.finditer(r"(?:至少|不少于)\s*(\d+)\s*集|(\d+)\s*集以上", text):
        pq.min_episodes = int(_num(m.group(1) or m.group(2)))
    for m in re.finditer(rf"{SCORE_CUE}\s*(?:of\s*)?(?:at least|above|over|>=|>|of|≥|no (?:lower|less) than)?\s*"
                         rf"{SCORE_VAL}\s*(\+|or (?:higher|more|above|better))?", low):
        v = float(m.group(1))
        if v >= 1 and re.search(r"at least|above|over|>=|>|\+|or higher|or more|or above|or better|≥|no (?:lower|less) "
                                r"than", m.group(0)):
            pq.min_score = v
    for m in re.finditer(rf"{SCORE_VAL}\s*(?:\+|or (?:higher|above))\s*{SCORE_CUE}", low):
        pq.min_score = float(m.group(1))
    for m in re.finditer(rf"{SCORE_CUE}\s*(?:of\s*)?(?:at most|no (?:higher|more) than|<=|≤)\s*{SCORE_VAL}|"
                         rf"{SCORE_CUE}\s*(?:of\s*)?{SCORE_VAL}\s*or (?:lower|less|below|worse)", low):
        pq.max_score = float(m.group(1) or m.group(2))
    m = re.search(rf"{SCORE_CUE}\s*(?:of\s*)?exactly\s*{SCORE_VAL}|\ba perfect 10\b", low)
    if m:
        pq.min_score = pq.max_score = float(m.group(1)) if m.group(1) else 10.0
    if pq.min_score is not None and pq.min_score > 10:
        pq.notes.append("Catalogue scores run from 1 to 10, so a minimum above 10 cannot be met.")
    _parse_years(low, text, pq)
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

    # ------------------------------------------------------------ studios (their names are not title mentions)
    studio_spans = _parse_studios(text, pq, lexicon)
    mention_text = text
    for a, b in studio_spans:
        mention_text = mention_text[:a] + " " * (b - a) + mention_text[b:]
    identification = _identification_question(low)

    # ------------------------------------------------------------ reference titles
    seen = set()
    for raw, start in _extract_reference_spans(text, quotes=not identification):
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
        if top.item_key in seen or (identification and top.match == "fuzzy"):
            continue    # in a plot description a quoted word ("quirks") is not a misspelt title
        if top.match == "fuzzy" and len(matches) > 1 and matches[1].score >= top.score - 2 \
                and matches[1].item_key != top.item_key:
            pq.ambiguous_titles.append({"mention": raw, "options": [
                {"item_key": m.item_key, "title": m.title} for m in matches[:3]]})
            continue
        seen.add(top.item_key)
        pq.reference_titles.append({"item_key": top.item_key, "title": top.title, "match": top.match,
                                    "mention": raw})
    if not pq.reference_titles and not identification:
        # bare mentions without a cue ("is Monster good?"); a long plot description names characters and places,
        # not titles, and a tag word ("School", "Romance") is a tag even when a title has that name
        for m, (a, b) in titles.find_mentions(mention_text):
            if lexicon.is_tag_phrase(text[a:b]):
                continue
            if m.item_key not in seen:
                seen.add(m.item_key)
                pq.reference_titles.append({"item_key": m.item_key, "title": m.title, "match": m.match,
                                            "mention": m.alias})
    ref_spans = [r["mention"].lower() for r in pq.reference_titles]

    # ------------------------------------------------------------ tags (negation / soft markers)
    tag_text = re.sub(r"(?<=\w)-(?=(?:tagged|themed|genre|style|flavou?red)\b)", " ", text, flags=re.IGNORECASE)
    t_norm = norm(tag_text)
    for start, end, ns, val in lexicon.find(tag_text):
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

    for k in ("include_tags", "exclude_tags", "soft_tags", "soft_avoid_tags"):
        setattr(pq, k, [list(t) for t in dict.fromkeys(tuple(t) for t in getattr(pq, k))])

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
    scene_words = re.search(r"which episode|what episode|the scene|scene where|scene in which|哪一集|第几集|名场面|场景",
                            low)
    pq.unavailable = _unavailable(low)
    if pq.unavailable == ["video timecodes"] and (scene_words or identification):
        pq.unavailable = []     # a scene question that also asks for the time: answered with the time marked unknown
    if re.search(r"screenshot|this (?:image|picture|frame|screen ?cap)|attached image|截图|这张图|图片里", low):
        pq.intent = "screenshot_lookup"
    elif pq.unavailable:
        pq.intent = "unavailable_data"
    elif pq.budget_minutes and re.search(r"plan|schedule|weekend|marathon|binge|hours|watch list|watchlist|"
                                         r"计划|周末|安排|小时", low):
        pq.intent = "viewing_plan"
    elif scene_words or (identification and not re.search(r"\bsequel (?:to|of)|\bprequel (?:to|of)|watch order", low)):
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
    if pq.year_from is not None and pq.year_to is not None and pq.year_from > pq.year_to:
        pq.needs_clarification = True
        pq.clarification = (f"You asked for titles that began airing in {pq.year_from} or later but in {pq.year_to} or "
                            "earlier. Which limit should I keep?")
    if pq.min_score is not None and pq.max_score is not None and pq.min_score > pq.max_score:
        pq.needs_clarification = True
        pq.clarification = (f"You asked for a score of at least {pq.min_score} but at most {pq.max_score}. "
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
                    ("min_score", pq.min_score), ("max_score", pq.max_score), ("year_from", pq.year_from),
                    ("year_to", pq.year_to), ("types", pq.types), ("include_tags", pq.include_tags),
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
    t = re.sub(rf"{SCORE_CUE}\s*(?:of\s*)?(?:at least|at most|above|over|exactly)?\s*\d{{1,2}}(?:\.\d+)?\+?"
               r"(?:\s*(?:or (?:higher|more|above|lower|less))|\s*out of 10)?", " ", t, flags=re.IGNORECASE)
    if pq.year_from is not None or pq.year_to is not None:
        t = re.sub(r"(?:began|begun|started|start(?:ing)?|aired|airing|released|premiered)?\s*(?:airing\s*)?"
                   r"(?:in|since|after|before|from|until|between)?\s*(?:19|20)\d{2}\s*(?:or (?:later|earlier|after|"
                   r"before|newer|older)|onwards?|and (?:19|20)\d{2})?", " ", t, flags=re.IGNORECASE)
    t = re.sub(rf"(?:{STRONG_STUDIO_CUE})\s+\S.*?{STUDIO_STOP}", " ", t, flags=re.IGNORECASE)
    t = re.sub(rf"{NUM}\s*(?:hours?|hrs?|minutes?|mins?)", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:i want|i'd like|looking for|recommend(?: me)?|give me|find me|suggest|can you|please|"
               r"something|anime|show|series)\b", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip(" ,.;!?")
