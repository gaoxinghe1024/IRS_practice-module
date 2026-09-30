"""Recommendation reasons, computed from data (never written by an LLM).

Each reason is {"code", "params", "text"}; `text` is the default English
rendering and `code` lets the UI localise later. All templates live here.
"""

from __future__ import annotations

import numpy as np

from app.core.text import norm

TEMPLATES = {
    "shared_tags": "Shares {ns}: {tags}",
    "same_demographic": "Also {tags}",
    "same_studio": "Same studio: {tags}",
    "similar_length": "Similar length ({a} vs {b} episodes)",
    "missing_dims": "Not compared (unknown): {dims}",
    "low_coverage": "Only part of the metadata could be compared",
    "profile_tags": "Matches this collection's {ns}: {tags}",
    "profile_length": "Close to the typical length in this collection",
    "filters_met": "Meets your filters: {filters}",
    "icf_support": "Because you rated {title} {rating}/10 (item similarity {sim}, {co} co-raters)",
    "icf_support_short": "Liked by people who rated {title} like you did",
    "mf_support": "Predicted from your ratings with matrix factorisation",
    "content_profile": "Matches your saved / liked {tags}",
    "popular_quality": "Popular and well rated (exploration)",
    "novel": "A less-watched title (novelty {nov})",
    "fresh": "Adds {tags}, new for your history",
    "list_role": "Keeps the list varied (list score drops by {delta} without it; explanatory estimate)",
    "retrieved": "Matched your description in {kind}",
    "plan_budget": "{minutes} min of your {budget} min budget",
    "plan_prereq": "Includes prerequisite {title} first",
}

DIM_LABEL = {"genre": "genres", "theme": "themes", "demographic": "demographic", "studio": "studio", "length": "length"}


def reason(code: str, **params) -> dict:
    return {"code": code, "params": params, "text": TEMPLATES[code].format(**params)}


def similarity_reasons(snap, target: int, cand: int, available: dict, sims: dict, pos: int) -> list[dict]:
    out = []
    for ns in ("genre", "theme"):
        tm = snap.tags.get(ns)
        if tm is not None and available[ns][pos]:
            shared = sorted(set(tm.tags_of(target)) & set(tm.tags_of(cand)))
            if shared:
                out.append(reason("shared_tags", ns=DIM_LABEL[ns], tags=", ".join(shared[:4])))
    for ns, code in (("demographic", "same_demographic"), ("studio", "same_studio")):
        tm = snap.tags.get(ns)
        if tm is not None and available[ns][pos]:
            shared = sorted(set(tm.tags_of(target)) & set(tm.tags_of(cand)))
            if shared:
                out.append(reason(code, tags=", ".join(shared[:3])))
    if available["length"][pos] and sims["length"][pos] >= 0.6:
        out.append(reason("similar_length", a=int(snap.episodes[cand]), b=int(snap.episodes[target])))
    missing = [DIM_LABEL[d] for d in ("genre", "theme", "demographic", "studio", "length") if not available[d][pos]]
    if missing:
        out.append(reason("missing_dims", dims=", ".join(missing)))
    return out


def profile_reasons(snap, profile, cand: int) -> list[dict]:
    out = []
    for ns in ("genre", "theme", "demographic", "studio"):
        tm = snap.tags.get(ns)
        f = profile.tag_freq.get(ns)
        if tm is None or f is None:
            continue
        hits = [tm.vocab[c] for c in tm.cols_of(cand) if f[c] > 0]
        if hits:
            hits.sort(key=lambda t: -f[tm.col[norm(t)]])
            out.append(reason("profile_tags", ns=DIM_LABEL[ns], tags=", ".join(hits[:4])))
    if profile.log_eps_mean is not None and np.isfinite(snap.episodes[cand]):
        if abs(np.log1p(snap.episodes[cand]) - profile.log_eps_mean) <= (profile.log_eps_sd or 0) + 0.5:
            out.append(reason("profile_length"))
    return out
