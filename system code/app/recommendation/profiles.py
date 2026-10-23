"""Preference profiles built from a specific collection or from a user's visible history."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.recommendation.features import PROFILE_NAMESPACES, CatalogSnapshot

NEGATIVE_ABSOLUTE = 5.0       # ratings <= 5 are negative signals
NEGATIVE_BELOW_MEAN = 1.5     # ...and so is anything more than 1.5 below the user's own mean


def is_negative(rating: float, user_mean: float | None) -> bool:
    """A rating is a negative signal when it is low in absolute terms or clearly below the
    user's own average (an 8/10 from a user who averages 8.4 is still positive)."""
    if rating <= NEGATIVE_ABSOLUTE:
        return True
    return user_mean is not None and rating < user_mean - NEGATIVE_BELOW_MEAN


@dataclass
class CollectionProfile:
    items: list[int]                                   # items that contributed (positive seeds)
    skipped: list[tuple[int, str]]                     # items not used as positive seeds, with reason
    tag_freq: dict[str, np.ndarray]                    # normalised frequency per namespace vocab
    ns_support: dict[str, int]                         # items with >= 1 tag in namespace
    log_eps_mean: float | None
    log_eps_sd: float | None
    size: int

    @property
    def empty(self) -> bool:
        return not self.items

    def top_tags(self, snap: CatalogSnapshot, ns: str, k: int = 5) -> list[dict]:
        f = self.tag_freq.get(ns)
        if f is None or not f.any():
            return []
        order = np.argsort(-f, kind="stable")[:k]
        return [{"tag": snap.tags[ns].vocab[c], "share": round(float(f[c]), 3)} for c in order if f[c] > 0]


def build_collection_profile(snap: CatalogSnapshot, items: list[int], ratings: dict[int, float] | None = None,
                             user_mean: float | None = None, excluded: set[int] | None = None) -> CollectionProfile:
    """Profile of the given collection items. Items the user rated low (is_negative) or
    explicitly excluded are not used as positive seeds."""
    use, skipped = [], []
    for i in items:
        if excluded and i in excluded:
            skipped.append((i, "not_interested"))
        elif ratings and i in ratings and is_negative(ratings[i], user_mean):
            skipped.append((i, "rated_low"))
        else:
            use.append(i)
    tag_freq: dict[str, np.ndarray] = {}
    support: dict[str, int] = {}
    for ns in PROFILE_NAMESPACES:
        tm = snap.tags.get(ns)
        if tm is None or not use:
            continue
        sub = tm.X[use]
        has = np.asarray(sub.sum(axis=1)).ravel() > 0
        support[ns] = int(has.sum())
        tag_freq[ns] = (np.asarray(sub.sum(axis=0)).ravel() / support[ns]) if support[ns] else \
            np.zeros(len(tm.vocab))
    eps = snap.episodes[use] if use else np.zeros(0)
    eps = eps[np.isfinite(eps)]
    lm = float(np.log1p(eps).mean()) if len(eps) else None
    ls = float(np.log1p(eps).std()) if len(eps) else None
    return CollectionProfile(use, skipped, tag_freq, support, lm, ls, len(use))


def affinity(snap: CatalogSnapshot, profile: CollectionProfile, candidates: np.ndarray
             ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """affinity_d(j) = sum_{t in tags_d(j)} profile_d(t) / |tags_d(j)|; plus length fit.
    Returns (values, available) per dimension; unavailable values are NaN."""
    vals, avail = {}, {}
    for ns in PROFILE_NAMESPACES:
        tm = snap.tags.get(ns)
        f = profile.tag_freq.get(ns)
        if tm is None or f is None or profile.ns_support.get(ns, 0) == 0:
            vals[ns] = np.full(len(candidates), np.nan)
            avail[ns] = np.zeros(len(candidates), dtype=bool)
            continue
        sizes = tm.sizes[candidates]
        s = np.asarray(tm.X[candidates] @ f).ravel()
        ok = sizes > 0
        v = np.full(len(candidates), np.nan)
        v[ok] = s[ok] / sizes[ok]
        vals[ns], avail[ns] = v, ok
    e = snap.episodes[candidates]
    ok = np.isfinite(e) & (profile.log_eps_mean is not None)
    v = np.full(len(candidates), np.nan)
    if profile.log_eps_mean is not None:
        sd2 = (profile.log_eps_sd or 0.0) ** 2 + 0.25
        v[ok] = np.exp(-((np.log1p(e[ok]) - profile.log_eps_mean) ** 2) / (2 * sd2))
    vals["length"], avail["length"] = v, ok
    return vals, avail


@dataclass
class UserState:
    """Everything the online recommenders may see about one application user."""

    user_id: int | None
    profile_version: int
    ratings: dict[int, float] = field(default_factory=dict)          # catalogue idx -> rating
    collections: dict[int, list[int]] = field(default_factory=dict)  # collection id -> items
    progress: dict[int, tuple[str, int]] = field(default_factory=dict)
    excluded_items: set[int] = field(default_factory=set)
    excluded_tags: list[tuple[str, str]] = field(default_factory=list)
    liked_tags: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rating_mean(self) -> float | None:
        return float(np.mean(list(self.ratings.values()))) if self.ratings else None

    def saved_items(self) -> set[int]:
        out: set[int] = set()
        for items in self.collections.values():
            out.update(items)
        return out

    def items_with_status(self, statuses: list[str]) -> set[int]:
        return {i for i, (st, _) in self.progress.items() if st in statuses}

    def history_items(self) -> set[int]:
        return set(self.ratings) | self.saved_items() | set(self.progress)

    def history_tags(self, snap: CatalogSnapshot, namespaces=("genre", "theme")) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for i in self.history_items():
            for ns in namespaces:
                tm = snap.tags.get(ns)
                if tm is not None:
                    out.update((ns, t) for t in tm.tags_of(i))
        return out

    def positive_items(self) -> list[int]:
        """Rated items that are not negative signals, plus saved items (never excluded or low-rated ones)."""
        mu = self.rating_mean
        low = {i for i, r in self.ratings.items() if is_negative(r, mu)}
        pos = set(self.ratings) - low
        pos |= (self.saved_items() - low)
        return sorted(pos - self.excluded_items)
