"""The eight fixed algorithm-2 features (one builder for training, evaluation and serving).

| # | name                 | definition                                                                       |
|---|----------------------|----------------------------------------------------------------------------------|
| 1 | rating_norm          | catalogue_score / 10 (the public catalogue score, not a personal rating)         |
| 2 | hits                 | |namespaced genre/theme tags of the candidate  ∩  profile top-10 tags|            |
| 3 | genre_affinity       | mean over the candidate's genres of freq_genre(tag)                              |
| 4 | theme_affinity       | same for themes                                                                  |
| 5 | demographic_affinity | same for demographics                                                            |
| 6 | studio_affinity      | same for studios                                                                 |
| 7 | length_affinity      | exp(-|log1p(episodes) - log1p(median visible episodes)|)                          |
| 8 | max_tag_similarity   | max over h in H of 0.5 * J(genres_i, genres_h) + 0.5 * J(themes_i, themes_h)    |

Missing values are NaN (LightGBM handles them natively): an affinity is NaN when the candidate has no
value in that namespace or no item of H has one; length is NaN when the candidate's or every visible
episode count is unknown; max_tag_similarity is NaN only for an empty H. Jaccard of two empty sets is 0
(missing tags are compared as empty sets). Features 3-8 are this experiment's implementation choice;
the slides define only the formula inputs (rating and hits).
"""

from __future__ import annotations

import numpy as np

from app.recommendation.algorithm2.catalog import Alg2Catalog
from app.recommendation.algorithm2.profiles import HIT_NAMESPACES, Alg2Profile

FEATURE_VERSION = "alg2-features-v1"
FEATURE_NAMES = ["rating_norm", "hits", "genre_affinity", "theme_affinity", "demographic_affinity",
                 "studio_affinity", "length_affinity", "max_tag_similarity"]
FORMULA_FEATURES = ["rating_norm", "hits"]
AFFINITY_NS = {"genre_affinity": "genre", "theme_affinity": "theme", "demographic_affinity": "demographic",
               "studio_affinity": "studio"}


def feature_schema() -> dict:
    return {"version": FEATURE_VERSION, "names": FEATURE_NAMES, "dtype": "float32", "missing": "NaN",
            "definitions": {
                "rating_norm": "catalogue_score / 10 (public catalogue score; never a personal rating)",
                "hits": "count of the candidate's namespaced genre/theme tags among the profile's top-10 "
                        "namespaced tags (frequency desc, ties by label asc)",
                "genre_affinity": "mean of profile frequency over the candidate's genres; NaN if candidate has none "
                                  "or no visible item has a genre",
                "theme_affinity": "as genre_affinity for themes",
                "demographic_affinity": "as genre_affinity for demographics",
                "studio_affinity": "as genre_affinity for studios",
                "length_affinity": "exp(-|log1p(episodes) - log1p(median visible episodes)|); NaN if unknown",
                "max_tag_similarity": "max over visible items of 0.5*Jaccard(genres) + 0.5*Jaccard(themes); "
                                      "empty union -> 0; NaN only for an empty history"},
            "profile": "frequencies are counts over the visible history H divided by |H|"}


def compute_features(cat: Alg2Catalog, prof: Alg2Profile, cand: np.ndarray) -> np.ndarray:
    cand = np.asarray(cand, dtype=np.int64)
    out = np.full((len(cand), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    if len(cand) == 0:
        return out.astype(np.float32)
    out[:, 0] = cat.score[cand] / 10.0
    hits = np.zeros(len(cand))
    for ns in HIT_NAMESPACES:
        v = prof.top_vec.get(ns)
        if v is not None and v.any():
            hits += np.asarray(cat.X[ns][cand] @ v).ravel()
    out[:, 1] = hits
    for j, name in enumerate(FEATURE_NAMES[2:6], start=2):
        ns = AFFINITY_NS[name]
        if prof.known.get(ns, 0) == 0:
            continue
        size = cat.sizes[ns][cand].astype(np.float64)
        s = np.asarray(cat.X[ns][cand] @ prof.freq[ns]).ravel()
        ok = size > 0
        out[ok, j] = s[ok] / size[ok]
    if prof.median_eps is not None:
        e = cat.episodes[cand]
        ok = np.isfinite(e)
        out[ok, 6] = np.exp(-np.abs(np.log1p(e[ok]) - np.log1p(prof.median_eps)))
    if prof.size:
        out[:, 7] = max_tag_similarity(cat, prof.history, cand)
    return out.astype(np.float32)


def _jaccard(cat: Alg2Catalog, ns: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Xa, Xb = cat.sig_X[ns][a], cat.sig_X[ns][b]
    inter = np.asarray((Xa @ Xb.T).todense(), dtype=np.float32)
    union = cat.sig_size[ns][a][:, None] + cat.sig_size[ns][b][None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def max_tag_similarity(cat: Alg2Catalog, history: np.ndarray, cand: np.ndarray, block: int = 4096) -> np.ndarray:
    hs = np.unique(cat.sig_id[history])
    cs, inv = np.unique(cat.sig_id[cand], return_inverse=True)
    best = np.zeros(len(cs), dtype=np.float32)
    for start in range(0, len(cs), block):
        part = cs[start:start + block]
        sim = 0.5 * _jaccard(cat, "genre", part, hs) + 0.5 * _jaccard(cat, "theme", part, hs)
        best[start:start + block] = sim.max(axis=1)
    return best[inv]
