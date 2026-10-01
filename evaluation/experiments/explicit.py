"""Explicit-rating diagnostic (MAE / RMSE) on the explicit protocol (per-user random 80/10/10).

Models are trained on explicit-train only (early stopping on explicit-validation) and scored on a
seeded sample of explicit-test users; every method predicts the same (user, item) pairs on the
same 1-10 scale. ICF predictions exist only where the item has rated neighbours; coverage is
reported, and the "with fallback" numbers use the user's train mean where ICF has no support.
This protocol is separate from the ranking protocol; its numbers are never merged into the
Hits/MRR tables as if they came from the same experiment.
"""

from __future__ import annotations

import time

import numpy as np

from evaluation.common import Run, md_table
from evaluation.metrics import bootstrap_ci, mae, rmse
from evaluation.protocols import load_context
from pipelines.split import load_split


def run(settings, sample_users_n: int | None = None, tag=None, **_) -> dict:
    ctx = load_context(settings, load_models=False)
    rec = settings.recommendation
    sample_users_n = sample_users_n or (20000 if settings.mode == "real" else None)
    run = Run(settings, "explicit", {"sample_test_users": sample_users_n, "tag": tag})
    tr = load_split(settings, "explicit", "train", ctx.version)
    va = load_split(settings, "explicit", "val", ctx.version)
    te = load_split(settings, "explicit", "test", ctx.version)
    items = np.unique(tr.item_idx.to_numpy())
    users = np.unique(tr.offline_user_id.to_numpy())
    cat_to_cf = np.full(ctx.snap.n, -1, dtype=np.int64)
    cat_to_cf[items] = np.arange(len(items))
    uidx = {int(u): n for n, u in enumerate(users)}

    def enc(df):
        ok = np.isin(df.offline_user_id.to_numpy(), users) & (cat_to_cf[df.item_idx.to_numpy()] >= 0)
        d = df[ok]
        return (np.array([uidx[int(u)] for u in d.offline_user_id]), cat_to_cf[d.item_idx.to_numpy()],
                d.rating.to_numpy(np.float64), int((~ok).sum()))

    tu, ti, tr_r, _ = enc(tr)
    vu, vi, vr, _ = enc(va)
    test_users = np.unique(te.offline_user_id)
    if sample_users_n and sample_users_n < len(test_users):
        test_users = np.sort(np.random.default_rng(5).choice(test_users, sample_users_n, replace=False))
    te_s = te[te.offline_user_id.isin(set(test_users.tolist()))]
    eu, ei, er, dropped = enc(te_s)
    run.notes.append(f"{dropped} test pairs dropped: user or item absent from explicit-train (cold pairs)")
    glob = float(tr_r.mean())
    user_mean = np.bincount(tu, weights=tr_r, minlength=len(users)) / np.maximum(1, np.bincount(tu, minlength=len(users)))
    item_mean = np.bincount(ti, weights=tr_r, minlength=len(items)) / np.maximum(1, np.bincount(ti, minlength=len(items)))
    preds: dict[str, np.ndarray] = {"global_mean": np.full(len(er), glob), "user_mean": user_mean[eu],
                                    "item_mean": item_mean[ei]}
    timings = {}

    # ---- ICF (explicit train)
    from app.recommendation.recall.icf import ICFScorer, fit_icf

    t0 = time.time()
    icf_model = fit_icf(tu, ti, tr_r, len(users), len(items), min_co_rated=int(rec["icf"].get("min_co_rated", 3)),
                        shrinkage=float(rec["icf"].get("shrinkage", 10)), top_n=int(rec["icf"].get("neighbors", 150)),
                        block_size=int(rec["icf"].get("block_size", 1024)))
    icf = ICFScorer(icf_model)                                  # the exact formula of the brief (no damping / prior)
    served = (float(rec["icf"].get("prediction_damping", 0.0)), float(rec["icf"].get("user_mean_prior_weight", 0.0)))
    icf_served = ICFScorer(icf_model, damping=served[0], prior_weight=served[1]) if any(served) else None
    run.notes.append(f"icf = exact formula; icf_served = served configuration (damping {served[0]}, "
                     f"user-mean prior weight {served[1]})" if icf_served else "served ICF uses the exact formula")
    timings["icf_fit_s"] = round(time.time() - t0, 1)
    p_icf = np.full(len(er), np.nan)
    p_icf_served = np.full(len(er), np.nan)
    order = np.argsort(eu, kind="stable")
    tr_order = np.argsort(tu, kind="stable")
    tr_starts = np.searchsorted(tu[tr_order], np.arange(len(users) + 1))
    for u in np.unique(eu):
        sel = order[np.searchsorted(eu[order], u):np.searchsorted(eu[order], u, side="right")]
        hist = tr_order[tr_starts[u]:tr_starts[u + 1]]
        pr = icf.predict(ti[hist], tr_r[hist], user_mean=float(user_mean[u]), n_ratings=len(hist))
        lookup = dict(zip(pr.items.tolist(), pr.predicted.tolist(), strict=True))
        p_icf[sel] = [lookup.get(int(i), np.nan) for i in ei[sel]]
        if icf_served is not None:
            ps = icf_served.predict(ti[hist], tr_r[hist], user_mean=float(user_mean[u]), n_ratings=len(hist))
            lk = dict(zip(ps.items.tolist(), ps.predicted.tolist(), strict=True))
            p_icf_served[sel] = [lk.get(int(i), np.nan) for i in ei[sel]]
    icf_cov = float(np.isfinite(p_icf).mean())
    preds["icf_with_fallback"] = np.where(np.isfinite(p_icf), p_icf, user_mean[eu])
    if icf_served is not None:
        preds["icf_served_with_fallback"] = np.where(np.isfinite(p_icf_served), p_icf_served, user_mean[eu])

    # ---- MF / NCF (explicit train, early stop on explicit validation)
    from app.recommendation.recall.matrix_factorization import fit_mf

    t0 = time.time()
    mf = fit_mf(tu, ti, tr_r, len(users), len(items), val=(vu, vi, vr), seed=0, **rec.get("mf", {}))
    timings["mf_fit_s"] = round(time.time() - t0, 1)
    preds["mf"] = np.clip(mf.predict_pairs(eu, ei), 1, 10)
    from app.recommendation.recall.ncf import fit_ncf, torch_available

    ncf_meta = None
    if torch_available():
        t0 = time.time()
        ncf = fit_ncf(tu, ti, tr_r, len(users), len(items), val=(vu, vi, vr), seed=0, **rec.get("ncf", {}))
        timings["ncf_fit_s"] = round(time.time() - t0, 1)
        preds["ncf_explicit"] = ncf.predict_pairs(eu, ei)
        ncf_meta = {"epochs_run": ncf.params.get("epochs_run"), "best_val_rmse": ncf.params.get("best_val_rmse"),
                    "history": ncf.history}
    else:
        run.notes.append("NCF not run: torch (ml extra) not installed")

    rows, metrics = [], {"test_pairs": int(len(er)), "test_users": int(len(np.unique(eu))), "icf_coverage": icf_cov,
                         "timings": timings, "mf": {"params": mf.params, "history": mf.history}, "ncf": ncf_meta}
    icf_mask = np.isfinite(p_icf)
    for name, p in preds.items():
        err = np.abs(p - er)
        metrics[name] = {"mae": bootstrap_ci(err.tolist()), "rmse": round(rmse(p, er), 5)}
        row = {"method": name, "mae": round(mae(p, er), 5), "rmse": round(rmse(p, er), 5),
               "mae_on_icf_supported": round(mae(p[icf_mask], er[icf_mask]), 5) if icf_mask.any() else None}
        rows.append(row)
    rows.append({"method": "icf_supported_only", "mae": round(mae(p_icf[icf_mask], er[icf_mask]), 5),
                 "rmse": round(rmse(p_icf[icf_mask], er[icf_mask]), 5),
                 "mae_on_icf_supported": round(mae(p_icf[icf_mask], er[icf_mask]), 5)})
    run.comparison(rows)
    for n in range(min(2000, len(er))):
        run.case({"user": int(eu[n]), "item": int(ei[n]), "rating": float(er[n]),
                  **{k: (None if not np.isfinite(v[n]) else round(float(v[n]), 3)) for k, v in preds.items()}})
    report = "\n".join([
        f"# Explicit-rating diagnostic (MAE / RMSE) - {settings.mode} data", "",
        f"Dataset `{ctx.version}`, explicit protocol (per-user random 80/10/10). Test pairs scored: {len(er):,} from "
        f"{len(np.unique(eu)):,} users. All methods predict the same pairs on the 1-10 scale.", "",
        md_table(rows, ["method", "mae", "rmse", "mae_on_icf_supported"]), "",
        f"ICF has neighbour support for {icf_cov:.1%} of test pairs; `icf_with_fallback` uses the user's train mean "
        "elsewhere and `icf_supported_only` restricts to supported pairs (not directly comparable with full-set rows).",
        "", "Training times: " + ", ".join(f"{k}={v}s" for k, v in timings.items()), "",
        "MAE/RMSE measure rating prediction, not ranking quality; the homepage experiment reports ranking metrics on "
        "the separate ranking protocol.", ""] + [f"- note: {n}" for n in run.notes]) + "\n"
    return run.finish(metrics, report, {"explicit_train_rows": int(len(tr_r))})
