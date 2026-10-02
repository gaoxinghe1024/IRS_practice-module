"""Algorithm-2 collection recommendation service (called by app/services/recommend.collection).

Uses the same catalogue, profile, rule engine, features and scorers as the offline experiments:
    effective collection H -> profile(H) -> strict rules (pool: served + catalogue score; user filters;
    collection items, completed titles and the user's exclusions removed) -> scorer -> Top-k.
The default scorer comes from the frozen run's decision (configs: recommendation.collection.scorer and
recommendation.collection.algorithm2_run). If the configured bundle is missing or incompatible, the
formula is used and the status says so; the user's filters are never relaxed by a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.recommendation.algorithm2.bundle import BundleMismatch, load_scorers
from app.recommendation.algorithm2.catalog import DEFAULT_SUPPLEMENT, Alg2Catalog, catalog_from_settings
from app.recommendation.algorithm2.profiles import Alg2Profile, build_profile
from app.recommendation.algorithm2.scorers import (
    CORE_IDS,
    LABELS,
    FormulaScorer,
    LegacyScorer,
    QueryContext,
    top_k,
)
from app.recommendation.constraints import ConstraintEngine, ConstraintSet, Eligibility

LEGACY_ALIASES = {"collection_profile_v1": "AUX_profile_v1", "manual_slides_v0": "AUX_slides_v0"}


@dataclass
class Alg2Result:
    items: np.ndarray
    scores: np.ndarray
    features: np.ndarray | None
    candidates: np.ndarray
    eligibility: Eligibility
    profile: Alg2Profile
    scorer_label: str
    scorer_id: str
    fallback: str | None = None


@dataclass
class Alg2Service:
    cat: Alg2Catalog
    engine: ConstraintEngine
    scorers: dict
    default: str
    status: str
    detail: str
    run_dir: str | None = None
    decision: dict = field(default_factory=dict)

    def resolve(self, requested: str | None) -> tuple[str, object, str | None]:
        label = LEGACY_ALIASES.get(requested or "", requested) or self.default
        if label in self.scorers:
            return label, self.scorers[label], None
        fb = self.default if self.default in self.scorers else "A0"
        return fb, self.scorers[fb], f"{requested} unavailable -> {fb}"

    def candidates(self, history, cs: ConstraintSet) -> tuple[np.ndarray, Eligibility]:
        cs = cs.copy(require_recommendable=False, require_scored=True)
        cs.exclude_items = set(cs.exclude_items) | {int(i) for i in history}
        elig = self.engine.evaluate(cs)
        m = self.cat.members
        return m[elig.mask[m]], elig

    def recommend(self, history, cs: ConstraintSet, scorer: str | None = None, k: int = 8) -> Alg2Result:
        prof = build_profile(self.cat, history)
        cand, elig = self.candidates(prof.history, cs)
        label, sc, fallback = self.resolve(scorer)
        ctx = QueryContext(self.cat, prof, cand)
        scores = sc.score(ctx) if len(cand) else np.zeros(0)
        top = top_k(scores, cand, self.cat.anime_id, k) if len(cand) else np.zeros(0, dtype=np.int64)
        feats = ctx._features[top] if ctx._features is not None else None
        return Alg2Result(cand[top], scores[top], feats, cand, elig, prof, label, getattr(sc, "scorer_id", label),
                          fallback)

    def public_status(self) -> dict:
        return {"status": self.status, "default": self.default, "label": LABELS.get(self.default.split("_seed")[0],
                                                                                   self.default),
                "detail": self.detail}


def build_service(snap, engine: ConstraintEngine, cc: dict, project_root: Path, log=None) -> Alg2Service:
    cat = catalog_from_settings(snap, project_root, cc.get("supplement", DEFAULT_SUPPLEMENT))
    requested = LEGACY_ALIASES.get(cc.get("scorer", "A0"), cc.get("scorer", "A0"))
    scorers: dict = {"A0": FormulaScorer("A0", 0.15), "AUX_slides_v0": LegacyScorer("AUX_slides_v0"),
                     "AUX_profile_v1": LegacyScorer("AUX_profile_v1")}
    run = cc.get("algorithm2_run")
    status, detail, decision = "ready", "", {}
    if run:
        run_dir = Path(run) if Path(run).is_absolute() else project_root / run
        try:
            fz_path = run_dir / "freeze_manifest.json"
            deploy_seed = json.loads(fz_path.read_text())["deploy_seed"] if fz_path.exists() else 42
            loaded, fz = load_scorers(run_dir, cat, [f"{f}_seed{deploy_seed}" for f in ("B0", "B1", "C0")], 1)
            scorers["A1"] = loaded["A1"]
            for fam in ("B0", "B1", "C0"):
                lab = f"{fam}_seed{deploy_seed}"
                if lab in loaded:
                    scorers[fam] = loaded[lab]
            dpath = run_dir / "decision.json"
            decision = json.loads(dpath.read_text()) if dpath.exists() else {}
            detail = f"frozen algorithm-2 run {run_dir.name}"
        except (BundleMismatch, ImportError, OSError, KeyError, ValueError) as exc:
            status = "degraded"
            detail = f"frozen run not loaded ({type(exc).__name__}: {str(exc)[:160]}); formula A0 in use"
            if log:
                log(detail)
    else:
        detail = "no frozen algorithm-2 run configured; formula A0 in use"
    default = requested
    if default not in scorers:
        fb = "A1" if "A1" in scorers else "A0"
        if status == "ready" and run:
            status = "degraded"
        detail += f"; configured scorer {requested} unavailable -> {fb}"
        default = fb
    detail += f"; default scorer {default}"
    return Alg2Service(cat, engine, scorers, default, status, detail, str(run) if run else None, decision)


def reasons_for(cat: Alg2Catalog, prof: Alg2Profile, i: int, reason_fn) -> list[dict]:
    """Data-derived reasons (same templates as before): shared profile tags per namespace, length."""
    labels = {"genre": "genres", "theme": "themes", "demographic": "demographic", "studio": "studio"}
    out = []
    for ns in ("genre", "theme", "demographic", "studio"):
        f = prof.freq.get(ns)
        if f is None:
            continue
        X = cat.X[ns]
        cols = X.indices[X.indptr[i]:X.indptr[i + 1]]
        hits = [c for c in cols if f[c] > 0]
        if hits:
            hits.sort(key=lambda c: (-f[c], cat.vocab[ns][c]))
            out.append(reason_fn("profile_tags", ns=labels[ns], tags=", ".join(cat.vocab[ns][c] for c in hits[:4])))
    e = cat.episodes[i]
    if prof.median_eps is not None and np.isfinite(e):
        if abs(np.log1p(e) - np.log1p(prof.median_eps)) <= 0.5:
            out.append(reason_fn("profile_length"))
    return out


def profile_summary(cat: Alg2Catalog, prof: Alg2Profile) -> dict:
    out = {ns: prof.shares(cat, ns, 5) for ns in ("genre", "theme", "demographic", "studio")}
    out["typical_episodes"] = None if prof.median_eps is None else round(float(prof.median_eps), 1)
    out["seed_items"] = int(prof.size)
    out["top_tags"] = prof.top_labels(cat)
    return out


__all__ = ["Alg2Service", "Alg2Result", "build_service", "reasons_for", "profile_summary", "CORE_IDS"]
