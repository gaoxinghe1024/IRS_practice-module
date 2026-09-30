"""Title / alias resolution: exact normalised alias, FTS prefix, then fuzzy (typos).

Resolution always returns catalogue item keys (never a model-invented title).
A sequel with a similar name is not accepted by embedding similarity; exact and
alias matches win, and ambiguous fuzzy matches are returned as alternatives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.core.text import nfkc, norm_title


def _fold(text: str) -> str:
    """Case-folded, whitespace-collapsed, punctuation KEPT ("Working'!!" is not "Working!!")."""
    return " ".join(nfkc(text).casefold().split())


@dataclass
class TitleMatch:
    item_key: str
    title: str
    match: str          # exact | alias | prefix | fuzzy
    score: float
    alias: str


class TitleResolver:
    def __init__(self, aliases: list[tuple[str, str, str]], titles: dict[str, str], popularity: dict[str, float],
                 served: set[str]):
        """aliases: (item_key, alias, kind)."""
        self.titles = titles
        self.popularity = popularity
        self.served = served
        self.exact: dict[str, list[tuple[str, str, str]]] = {}
        self.folded: dict[str, set[str]] = {}      # punctuation-preserving alias -> item keys
        self.choices: list[str] = []
        self.choice_rows: list[tuple[str, str, str]] = []
        for key, alias, kind in aliases:
            if key not in served:
                continue
            n = norm_title(alias)
            if not n:
                continue
            self.exact.setdefault(n, []).append((key, alias, kind))
            self.folded.setdefault(_fold(alias), set()).add(key)
            self.choices.append(n)
            self.choice_rows.append((key, alias, kind))
        self.max_words = max((len(c.split()) for c in self.choices), default=1)

    def _rank(self, rows: list[tuple[str, str, str]], match: str, score: float, text: str | None = None
              ) -> list[TitleMatch]:
        # several titles can share a normalised form (sequels that differ only in punctuation); the one whose
        # alias matches the text with its punctuation wins, then titles over aliases, then popularity
        literal = (self.folded.get(_fold(text), set()) | self.folded.get(_fold(text.rstrip("?.,;:")), set())
                   if text else set())
        seen, out = set(), []
        for key, alias, kind in sorted(rows, key=lambda r: (r[0] not in literal, r[2] != "title",
                                                            -self.popularity.get(r[0], 0), r[0])):
            if key in seen:
                continue
            seen.add(key)
            out.append(TitleMatch(key, self.titles.get(key, alias), "exact" if kind == "title" else match, score, alias))
        return out

    def resolve(self, text: str, limit: int = 5, fuzzy_cutoff: float = 86.0) -> list[TitleMatch]:
        n = norm_title(text)
        if not n:
            return []
        if n in self.exact:
            return self._rank(self.exact[n], "alias", 100.0, text)[:limit]
        try:
            from rapidfuzz import fuzz, process
        except ImportError:  # pragma: no cover - base dependency
            return []
        res = process.extract(n, self.choices, scorer=fuzz.WRatio, limit=limit * 4, score_cutoff=fuzzy_cutoff)
        rows = []
        for choice, score, idx in res:
            # WRatio favours partial matches; require the lengths to be comparable
            if len(choice) < 0.5 * len(n) or len(n) < 0.5 * len(choice):
                continue
            rows.append((self.choice_rows[idx], score))
        out, seen = [], set()
        for (key, alias, _kind), score in sorted(rows, key=lambda r: (-r[1], -self.popularity.get(r[0][0], 0), r[0][0])):
            if key not in seen:
                seen.add(key)
                out.append(TitleMatch(key, self.titles.get(key, alias), "fuzzy", float(score), alias))
        return out[:limit]

    def longest_prefix_mention(self, text: str) -> str | None:
        """The longest leading span of `text` that is exactly a catalogue alias (punctuation included), if any.

        Used after a cue such as "tell me about": titles like "Working'!!" or "Komi-san wa, Comyushou desu."
        contain the punctuation a clause splitter would cut at.
        """
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        for size in range(min(self.max_words, len(words)), 0, -1):
            span = text[words[0][0]:words[size - 1][1]]
            for cand in (span, span.rstrip("?.,;:")):
                if cand and norm_title(cand) in self.exact:
                    return cand
        return None

    def find_mentions(self, text: str, min_words: int = 1) -> list[tuple[TitleMatch, tuple[int, int]]]:
        """Exact alias mentions inside free text (longest match first, non-overlapping)."""
        words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        found, used = [], np.zeros(len(words), dtype=bool)
        for size in range(min(self.max_words, len(words)), min_words - 1, -1):
            for i in range(0, len(words) - size + 1):
                if used[i:i + size].any():
                    continue
                span = text[words[i][1]:words[i + size - 1][2]]
                n = norm_title(span)
                # a single lowercase word ("a monster hunting show") is not a title mention
                single_ok = size > 1 or (len(n) >= 5 and (span[:1].isupper() or not span.isascii()))
                if n in self.exact and single_ok:
                    m = self._rank(self.exact[n], "alias", 100.0, span)[0]
                    found.append((m, (words[i][1], words[i + size - 1][2])))
                    used[i:i + size] = True
        return found
