"""One constraint engine for catalogue browsing, collection recommendations, the
home feed, chat and the planner.

Every rule has an id, a source and a relaxable flag. Hard rules are evaluated
as vectorised masks over the catalogue snapshot; a missing value never counts
as satisfying a hard filter (unknown score fails "score >= 8", unknown episode
count fails "<= 12 episodes", unknown themes fail "no <theme>"). Relaxation
options are *suggestions* returned next to an empty strict result; they are
never applied without the user choosing them, and content / account exclusion
/ explicit negative-preference rules are never offered for relaxation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.core.text import norm
from app.recommendation.features import CatalogSnapshot

RULES: dict[str, dict] = {
    "content.explicit": {"relaxable": False, "label": "Content policy"},
    "pool.recommendable": {"relaxable": False, "label": "Has a catalogue score and genre"},
    "pool.scored": {"relaxable": False, "label": "Has a catalogue score"},
    "pool.planner_eligible": {"relaxable": False, "label": "Reliable episode count and duration"},
    "user.excluded_item": {"relaxable": False, "label": "You marked it not interested"},
    "user.excluded_tag": {"relaxable": False, "label": "You excluded this tag"},
    "dedupe.excluded_items": {"relaxable": False, "label": "Already saved / rated / watched"},
    "filter.min_score": {"relaxable": True, "label": "Minimum score"},
    "filter.max_episodes": {"relaxable": True, "label": "Maximum episodes"},
    "filter.min_episodes": {"relaxable": True, "label": "Minimum episodes"},
    "filter.types": {"relaxable": True, "label": "Type"},
    "filter.include_tags": {"relaxable": True, "label": "Required tags"},
    "filter.exclude_tags": {"relaxable": False, "label": "Excluded tags (your explicit request)"},
    "filter.keywords": {"relaxable": True, "label": "Keywords"},
    "filter.statuses": {"relaxable": True, "label": "Airing status"},
    "filter.max_total_minutes": {"relaxable": False, "label": "Fits the time budget"},
}


@dataclass
class ConstraintSet:
    min_score: float | None = None
    max_episodes: int | None = None
    min_episodes: int | None = None
    types: list[str] | None = None
    statuses: list[str] | None = None
    include_tags: list[tuple[str, str]] = field(default_factory=list)   # (namespace, value), all required
    exclude_tags: list[tuple[str, str]] = field(default_factory=list)   # request-level negative preference
    keywords: list[str] = field(default_factory=list)                   # contains-match on genre or title/alias
    max_total_minutes: float | None = None
    require_recommendable: bool = True
    require_scored: bool = False                                        # catalogue score known (algorithm 2 pool)
    require_planner_eligible: bool = False
    exclude_items: set[int] = field(default_factory=set)                # dedupe (saved / rated / reference)
    user_excluded_items: set[int] = field(default_factory=set)
    user_excluded_tags: list[tuple[str, str]] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)               # rule_id -> where it came from

    def active_rules(self) -> list[str]:
        r = ["content.explicit"]
        if self.require_recommendable:
            r.append("pool.recommendable")
        if self.require_scored:
            r.append("pool.scored")
        if self.require_planner_eligible:
            r.append("pool.planner_eligible")
        if self.user_excluded_items:
            r.append("user.excluded_item")
        if self.user_excluded_tags:
            r.append("user.excluded_tag")
        if self.exclude_items:
            r.append("dedupe.excluded_items")
        for rid, val in (("filter.min_score", self.min_score), ("filter.max_episodes", self.max_episodes),
                         ("filter.min_episodes", self.min_episodes), ("filter.types", self.types),
                         ("filter.statuses", self.statuses), ("filter.include_tags", self.include_tags),
                         ("filter.exclude_tags", self.exclude_tags), ("filter.keywords", self.keywords),
                         ("filter.max_total_minutes", self.max_total_minutes)):
            if val not in (None, [], set()):
                r.append(rid)
        return r

    def to_public(self) -> dict:
        return {"min_score": self.min_score, "max_episodes": self.max_episodes, "min_episodes": self.min_episodes,
                "types": self.types, "statuses": self.statuses,
                "include_tags": [list(t) for t in self.include_tags],
                "exclude_tags": [list(t) for t in self.exclude_tags], "keywords": self.keywords,
                "max_total_minutes": self.max_total_minutes, "sources": self.sources}

    def copy(self, **changes) -> ConstraintSet:
        import copy as _copy

        c = _copy.deepcopy(self)
        for k, v in changes.items():
            setattr(c, k, v)
        return c


@dataclass
class Eligibility:
    mask: np.ndarray
    rule_masks: dict[str, np.ndarray]     # True = passes that rule
    fail_counts: dict[str, int]
    unknown_tag_namespaces: dict[str, int]

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    def fail_reasons(self, i: int) -> list[str]:
        return [rid for rid, m in self.rule_masks.items() if not m[i]]


class ConstraintEngine:
    def __init__(self, snap: CatalogSnapshot):
        self.snap = snap
        self._kw_text: list[str] | None = None
        self._aliases: dict[int, list[str]] = {}
        self._kw_masks: dict[str, np.ndarray] = {}

    def set_aliases(self, aliases: dict[int, list[str]]) -> None:
        self._aliases = aliases
        self._kw_text = None
        self._kw_masks = {}

    def keyword_mask(self, keyword: str) -> np.ndarray:
        """Titles whose genre text or title / alias text contains the (normalised) keyword. Cached per keyword."""
        m = self._kw_masks.get(keyword)
        if m is None:
            texts = self._keyword_text()
            gtexts = self._genre_text
            m = np.array([(keyword in gtexts[i]) or (keyword in texts[i]) for i in range(self.snap.n)], dtype=bool)
            self._kw_masks[keyword] = m
        return m

    def _keyword_text(self) -> list[str]:
        if self._kw_text is None:
            snap = self.snap
            genre = snap.tags.get("genre")
            texts = []
            for i in range(snap.n):
                parts = [snap.titles[i], *self._aliases.get(i, [])]
                texts.append(norm(" | ".join(parts)))
            self._kw_text = texts
            self._genre_text = [norm(" | ".join(genre.tags_of(i))) if genre else "" for i in range(snap.n)]
        return self._kw_text

    def _tag_mask(self, ns: str, value: str) -> tuple[np.ndarray, np.ndarray]:
        """(has_tag, namespace_known) masks."""
        tm = self.snap.tags.get(ns)
        if tm is None:
            z = np.zeros(self.snap.n, dtype=bool)
            return z, z
        col = tm.col.get(norm(value))
        known = tm.sizes > 0
        if col is None:
            return np.zeros(self.snap.n, dtype=bool), known
        return np.asarray(tm.X[:, col].toarray()).ravel() > 0, known

    def evaluate(self, cs: ConstraintSet) -> Eligibility:
        snap = self.snap
        masks: dict[str, np.ndarray] = {}
        unknown_ns: dict[str, int] = {}
        masks["content.explicit"] = snap.in_catalog & ~snap.explicit
        if cs.require_recommendable:
            masks["pool.recommendable"] = snap.recommendable.copy()
        if cs.require_scored:
            masks["pool.scored"] = np.isfinite(snap.score)
        if cs.require_planner_eligible:
            masks["pool.planner_eligible"] = snap.planner_eligible.copy()
        if cs.user_excluded_items:
            m = np.ones(snap.n, dtype=bool)
            m[list(cs.user_excluded_items)] = False
            masks["user.excluded_item"] = m
        if cs.user_excluded_tags:
            m = np.ones(snap.n, dtype=bool)
            for ns, v in cs.user_excluded_tags:
                has, _known = self._tag_mask(ns, v)
                m &= ~has
            masks["user.excluded_tag"] = m
        if cs.exclude_items:
            m = np.ones(snap.n, dtype=bool)
            m[list(cs.exclude_items)] = False
            masks["dedupe.excluded_items"] = m
        if cs.min_score is not None:
            masks["filter.min_score"] = np.nan_to_num(snap.score, nan=-np.inf) >= cs.min_score
        if cs.max_episodes is not None:
            masks["filter.max_episodes"] = np.nan_to_num(snap.episodes, nan=np.inf) <= cs.max_episodes
        if cs.min_episodes is not None:
            masks["filter.min_episodes"] = np.nan_to_num(snap.episodes, nan=-np.inf) >= cs.min_episodes
        if cs.types:
            allowed = {t.casefold() for t in cs.types}
            masks["filter.types"] = np.array([(t or "").casefold() in allowed for t in snap.type])
        if cs.statuses:
            allowed = {t.casefold() for t in cs.statuses}
            masks["filter.statuses"] = np.array([(t or "").casefold() in allowed for t in snap.status])
        if cs.include_tags:
            m = np.ones(snap.n, dtype=bool)
            for ns, v in cs.include_tags:
                has, _ = self._tag_mask(ns, v)
                m &= has
            masks["filter.include_tags"] = m
        if cs.exclude_tags:
            m = np.ones(snap.n, dtype=bool)
            for ns, v in cs.exclude_tags:
                has, known = self._tag_mask(ns, v)
                unknown_ns[ns] = unknown_ns.get(ns, 0) + int((~known).sum())
                m &= known & ~has    # unknown namespace cannot be verified -> fails
            masks["filter.exclude_tags"] = m
        if cs.keywords:
            m = np.ones(snap.n, dtype=bool)
            for k in cs.keywords:                 # every keyword must match (AND)
                m &= self.keyword_mask(k)
            masks["filter.keywords"] = m
        if cs.max_total_minutes is not None:
            masks["filter.max_total_minutes"] = np.nan_to_num(snap.total_minutes, nan=np.inf) <= cs.max_total_minutes
        mask = np.ones(snap.n, dtype=bool)
        for m in masks.values():
            mask &= m
        fails = {rid: int((~m).sum()) for rid, m in masks.items()}
        return Eligibility(mask, masks, fails, unknown_ns)

    def suggest_relaxations(self, cs: ConstraintSet, k: int, max_options: int = 4) -> list[dict]:
        """Alternative filters that would yield candidates; each changes one relaxable rule."""
        base = self.evaluate(cs)
        if base.count >= k:
            return []
        options = []
        snap = self.snap

        def others_mask(skip: str) -> np.ndarray:
            m = np.ones(snap.n, dtype=bool)
            for rid, rm in base.rule_masks.items():
                if rid != skip:
                    m &= rm
            return m

        if cs.max_episodes is not None:
            om = others_mask("filter.max_episodes")
            eps = np.sort(snap.episodes[om & np.isfinite(snap.episodes)])
            if len(eps):
                target = eps[min(len(eps), k) - 1]
                new = int(math.ceil(target))
                if new > cs.max_episodes:
                    n_new = int((om & (np.nan_to_num(snap.episodes, nan=np.inf) <= new)).sum())
                    options.append({"rule_id": "filter.max_episodes", "change": {"max_episodes": new},
                                    "candidate_count": n_new,
                                    "reason": f"Raising the episode cap from {cs.max_episodes} to {new} gives "
                                              f"{n_new} eligible titles."})
        if cs.min_score is not None:
            om = others_mask("filter.min_score")
            sc = np.sort(snap.score[om & np.isfinite(snap.score)])[::-1]
            if len(sc):
                new = math.floor(float(sc[min(len(sc), k) - 1]) * 10) / 10
                if new < cs.min_score:
                    n_new = int((om & (np.nan_to_num(snap.score, nan=-np.inf) >= new)).sum())
                    options.append({"rule_id": "filter.min_score", "change": {"min_score": new},
                                    "candidate_count": n_new,
                                    "reason": f"Lowering the minimum score from {cs.min_score} to {new} gives "
                                              f"{n_new} eligible titles."})
        if cs.min_episodes is not None:
            om = others_mask("filter.min_episodes")
            n_new = int(om.sum())
            if n_new:
                options.append({"rule_id": "filter.min_episodes", "change": {"min_episodes": None},
                                "candidate_count": n_new, "reason": "Removing the minimum episode count."})
        if cs.keywords:
            for kw in cs.keywords:
                remaining = [x for x in cs.keywords if x != kw]
                n_new = self.evaluate(cs.copy(keywords=remaining)).count
                if n_new:
                    options.append({"rule_id": "filter.keywords", "change": {"keywords": remaining},
                                    "candidate_count": n_new, "reason": f'Dropping the keyword "{kw}".'})
        if cs.include_tags:
            for t in cs.include_tags:
                remaining = [x for x in cs.include_tags if x != t]
                n_new = self.evaluate(cs.copy(include_tags=remaining)).count
                if n_new:
                    options.append({"rule_id": "filter.include_tags",
                                    "change": {"include_tags": [list(x) for x in remaining]},
                                    "candidate_count": n_new, "reason": f'Not requiring the tag "{t[1]}".'})
        if cs.types:
            n_new = int(others_mask("filter.types").sum())
            if n_new:
                options.append({"rule_id": "filter.types", "change": {"types": None}, "candidate_count": n_new,
                                "reason": "Allowing any type."})
        if cs.statuses:
            n_new = int(others_mask("filter.statuses").sum())
            if n_new:
                options.append({"rule_id": "filter.statuses", "change": {"statuses": None},
                                "candidate_count": n_new, "reason": "Allowing any airing status."})
        if not options:
            # no single change helps: offer clearing every adjustable filter (still the user's choice)
            cleared = cs.copy(min_score=None, max_episodes=None, min_episodes=None, types=None, statuses=None,
                              include_tags=[], keywords=[])
            n_new = self.evaluate(cleared).count
            if n_new:
                options.append({"rule_id": "filter.*", "change": {"min_score": None, "max_episodes": None,
                                                                   "min_episodes": None, "types": None,
                                                                   "include_tags": [], "keywords": []},
                                "candidate_count": n_new,
                                "reason": "No single change is enough; clearing the adjustable filters gives "
                                          f"{n_new} eligible titles (content rules and your exclusions stay)."})
        options.sort(key=lambda o: (-min(o["candidate_count"], k), o["rule_id"]))
        return options[:max_options]


def alias_map(snap: CatalogSnapshot, aliases) -> dict[int, list[str]]:
    """Catalogue index -> alternative titles (an `aliases` table with item_key / alias columns)."""
    out: dict[int, list[str]] = {}
    for key, alias in zip(aliases["item_key"], aliases["alias"], strict=True):
        i = snap.index.get(key)
        if i is not None and alias != snap.titles[i]:
            out.setdefault(i, []).append(alias)
    return out


def build_engine(snap: CatalogSnapshot, aliases=None) -> ConstraintEngine:
    """The engine used by serving and by offline experiments (same keyword text, same rules)."""
    ce = ConstraintEngine(snap)
    if aliases is not None:
        ce.set_aliases(alias_map(snap, aliases))
    return ce
