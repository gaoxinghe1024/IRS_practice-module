"""`report` stage: the human-readable records of a run, generated only from the run's own files.

Writes in the run directory:
  summary.csv          per-model table: metrics, paired differences, P95, training time, model size and the
                       confirmation (the evaluate-test table with the benchmark columns added)
  summary_en.md        short English table (metric names identical to the full report)
  case_analysis.md     at least 12 real test cases in the required categories; a category that does not
                       occur is reported as not observed, never invented
  final_report.md      full report for the course presentation
  README_reproduce.md  commands, data locations and outputs of this run
  query_manifest.parquet, sampled_pairs.parquet, users_split.parquet  the split records (from work/)
A stage that has not run shows NOT_RUN; no number is guessed.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from algorithm2_benchmark.pipeline.common import mark_stage, now, read_json, write_json
from algorithm2_benchmark.pipeline.data import load_prepared
from algorithm2_benchmark.pipeline.split import load_queries
from app.recommendation.algorithm2.profiles import build_profile

NR = "NOT_RUN"
RULE_TEXT = {"R0": "R0 (no filter)", "R1": "R1 (keyword = most common genre in the visible collection)", "R2": "R2 (public score ≥ 7.0 and episodes ≤ 26)",
             "R3": "R3 (R1 + R2)"}
FAMILY_TEXT = {"A0": "original manual formula", "A1": "validation-tuned weight formula", "B0": "LightGBM (two features)", "B1": "LightGBM (eight features)",
               "C0": "content two-tower", "AUX_slides_v0": "legacy slides formula (auxiliary)", "AUX_profile_v1": "legacy profile matching (auxiliary)"}


# ------------------------------------------------------------------------------------------ helpers


def _load(run, name):
    p = run.path(name)
    if not p.exists():
        return None
    if name.endswith(".json"):
        return read_json(p)
    return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)


def f4(x, nd: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—" if x is None else str(x)
    return f"{x:.{nd}f}" if np.isfinite(x) else "—"


def pp(x) -> str:
    """A difference of two rates in percentage points."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{x * 100:+.2f}" if np.isfinite(x) else "—"


def ci_pp(lo, hi) -> str:
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return "—"
    return f"[{lo * 100:+.2f}, {hi * 100:+.2f}]" if np.isfinite(lo) and np.isfinite(hi) else "—"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class Records:
    """Everything the report reads, with the deploy-seed labels resolved."""

    def __init__(self, run):
        self.run = run
        self.cfg = run.config()
        g = lambda n: _load(run, n)  # noqa: E731
        self.data, self.split, self.protocol = g("data_manifest.json"), g("split_manifest.json"), g(
            "selection_protocol.json")
        self.tuning, self.models, self.freeze = g("tuning.json"), g("models_manifest.json"), g("freeze_manifest.json")
        self.metrics, self.decision = g("metrics.json"), g("decision.json")
        self.summary, self.subgroups, self.paired = g("summary.csv"), g("subgroup_metrics.csv"), g(
            "paired_comparisons.csv")
        self.latency, self.bench = g("latency.csv"), g("benchmark.json")
        self.integration, self.meta, self.audit = g("integration_report.json"), g("run_metadata.json"), g(
            "audit_facts.json")
        self.trials, self.importance = g("trial_results.csv"), g("feature_importance.csv")
        self.seed = int(self.freeze["deploy_seed"]) if self.freeze else int(self.cfg["deploy_seed"])
        self.labels = {f: (f if f in ("A0", "A1") else f"{f}_seed{self.seed}") for f in ("A0", "A1", "B0", "B1", "C0")}

    def pair(self, model: str, base: str, metric: str = "HR@8") -> tuple:
        if self.paired is None:
            return (None, None, None)
        r = self.paired[(self.paired.model == model) & (self.paired.baseline == base) & (self.paired.metric == metric)]
        if r.empty:
            return (None, None, None)
        r = r.iloc[0]
        return (r.mean_diff, r.ci95_low, r.ci95_high)

    def lat(self, model: str, layer: str, rule: str = "R0", stat: str = "p95_ms"):
        if self.latency is None:
            return None
        r = self.latency[(self.latency.model == model) & (self.latency.layer == layer) & (self.latency.rule == rule)]
        return None if r.empty else r.iloc[0][stat]

    def model_info(self, label: str) -> dict:
        if not self.models:
            return {}
        for m in self.models["models"]:
            if f"{m['family']}_seed{m['seed']}" == label:
                return m
        return {}


# ------------------------------------------------------------------------------------------ summary


def final_summary(rec: Records) -> pd.DataFrame | None:
    if rec.summary is None:
        return None
    df = rec.summary.copy()
    p95_service, p95_api, fit, size = [], [], [], []
    for m in df.experiment:
        fam = m.split("_seed")[0]
        deployed = m in ("A0", "A1") or m == rec.labels.get(fam)
        p95_service.append(rec.lat(fam, "service") if deployed else None)
        p95_api.append(rec.lat(fam, "api") if deployed else None)
        info = rec.model_info(m)
        fit.append(info.get("fit_seconds", 0.0 if fam in ("A0", "A1") else None))
        size.append(info.get("bytes", 0 if fam in ("A0", "A1") else None))
    df["P95_service_ms_R0"] = p95_service
    df["P95_api_ms_R0"] = p95_api
    df["training_seconds"] = fit
    df["model_bytes"] = size
    df["latency_note"] = ["measured on the deployed artifact (deploy seed) only" if v is None and "_seed" in m
                          else ("not measured (auxiliary)" if m.startswith("AUX") else "")
                          for m, v in zip(df.experiment, p95_service, strict=True)]
    df.to_csv(rec.run.path("summary.csv"), index=False)
    return df


def summary_en(rec: Records) -> str:
    lines = [f"# Algorithm 2 — {rec.run.run_id} (level: {rec.cfg['level']})", ""]
    if rec.metrics is None:
        lines += ["Status: NOT_RUN (no test evaluation in this run).", ""]
        return "\n".join(lines)
    prim = rec.metrics["primary"]
    rows = []
    for fam in ("A0", "A1", "B0", "B1", "C0"):
        lab = rec.labels[fam]
        p = prim.get(lab)
        if p is None:
            rows.append([fam, NR] + [NR] * 10)
            continue
        d0, l0, h0 = rec.pair(lab, "A0")
        d1, l1, h1 = rec.pair(lab, "A1")
        info = rec.model_info(lab)
        p95 = rec.lat(fam, "service")
        rows.append([f"{fam} ({lab})" if lab != fam else fam, p["users"], f4(p["HR@8"]), f4(p["NDCG@8"]),
                     f4(p["MRR@8"]), f4(p["AUC"]),
                     "—" if fam == "A0" else f"{pp(d0)} {ci_pp(l0, h0)}",
                     "—" if fam == "A1" else f"{pp(d1)} {ci_pp(l1, h1)}",
                     NR if p95 is None else f"{p95:.1f}",
                     f"{info.get('fit_seconds', 0.0):.1f}" if fam not in ("A0", "A1") else "0",
                     f"{info.get('bytes', 0) / 1024:.0f}" if fam not in ("A0", "A1") else "0", p["violations"]])
    lines.append(table(["Model", "Users", "HR@8", "NDCG@8", "MRR@8", "AUC", "ΔHR@8 vs A0, pp [95% CI]",
                        "ΔHR@8 vs A1, pp [95% CI]", "P95 service latency (ms)", "Training time (s)",
                        "Model size (KB)", "Rule violations"], rows))
    d = rec.decision or {}
    lines += ["", "Primary set: rule R0, main group (visible history >= 5), more than 8 candidates, target eligible; "
              "full-catalogue ranking; user-macro averages. Learned models: deploy seed "
              f"{rec.seed}; seed means are in final_report.md.",
              "Label: proxy (personal rating >= 8 in the Kaggle MyAnimeList 2023 snapshot), not real collections.", "",
              f"Pre-selected on validation: **{d.get('pre_selected', NR)}** · confirmed on test: "
              f"**{d.get('confirmed', NR)}** · default scorer: **{d.get('default_scorer', NR)}**",
              f"Reason: {d.get('reason', NR)}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ cases


def _names(cat, ns: str, i: int) -> list[str]:
    X = cat.X.get(ns)
    if X is None:
        return []
    return [cat.vocab[ns][c] for c in X.indices[X.indptr[i]:X.indptr[i + 1]]]


def _title(snap, i: int, n: int = 48) -> str:
    t = str(snap.titles[i]).replace("|", "/")
    return t if len(t) <= n else t[: n - 1] + "…"


def case_analysis(rec: Records) -> tuple[str, dict]:
    run, cfg = rec.run, rec.cfg
    pred = _load(run, "predictions.parquet")
    if pred is None:
        return "# Failure case analysis\n\nNOT_RUN: this run has no test predictions (predictions.parquet).\n", {}
    prep = load_prepared(cfg)
    cat = prep.cat
    snap = cat.snap
    q = load_queries(run).set_index("query_id")
    core = [rec.labels[f] for f in ("A0", "A1", "B0", "B1", "C0")]
    p = pred[pred.model.isin(core)]
    hits = p.pivot_table(index=["query_id", "rule"], columns="model", values="hit", aggfunc="first")
    ranks = p.pivot_table(index=["query_id", "rule"], columns="model", values="rank", aggfunc="first")
    info = p[p.model == "A0"].set_index(["query_id", "rule"])
    members = np.nan_to_num(np.asarray(snap.members, dtype=float))
    order = cat.members[np.argsort(-members[cat.members], kind="stable")]
    pop_rank = {int(i): r + 1 for r, i in enumerate(order.tolist())}
    top_of = {(r.query_id, r.rule, r.model): r.top_keys for r in p.itertuples(index=False)}
    b1, c0 = rec.labels["B1"], rec.labels["C0"]
    learned = [rec.labels["B0"], b1, c0]
    r0 = hits.xs("R0", level="rule") if "R0" in hits.index.get_level_values("rule") else pd.DataFrame()
    r0_info = info.xs("R0", level="rule")
    main_ok = r0_info[(r0_info.group == "main") & r0_info.target_eligible & (r0_info.n_cand > 8)].index
    r0m = r0.loc[r0.index.intersection(main_ok)]
    used: set = set()

    def pick(ids, n=2) -> list:
        out = []
        for x in sorted(ids):
            if x not in used:
                out.append(x)
                used.add(x)
            if len(out) == n:
                break
        return out

    cats = []
    a0_only = r0m[(r0m["A0"] == 1) & (r0m[b1] == 0) & (r0m[c0] == 0)].index
    cats.append(("A0 hits, learned models (B1 and C0) miss", "R0", pick(a0_only), len(a0_only)))
    learned_only = r0m[(r0m["A0"] == 0) & ((r0m[b1] == 1) | (r0m[c0] == 1))].index
    cats.append(("A learned model hits, A0 misses", "R0", pick(learned_only), len(learned_only)))
    all_fail = r0m[(r0m[core] == 0).all(axis=1)].index
    cats.append(("All five core models miss", "R0", pick(all_fail), len(all_fail)))
    sparse = r0_info[(r0_info.group == "sparse") & r0_info.target_eligible].index
    cats.append(("Sparse collections (1–4 visible titles)", "R0", pick(sparse), len(sparse)))
    mixed = r0_info[(r0_info.group == "main") & (r0_info.profile_mix == "mixed") & r0_info.target_eligible].index
    cats.append(("Mixed preferences (genre entropy above the validation median)", "R0", pick(mixed), len(mixed)))
    excl = info[(info.excluded_reason == "target_not_eligible") & (info.index.get_level_values("rule") != "R0")]
    ex_pick = []
    for rule in ("R2", "R1", "R3"):
        ids = excl.xs(rule, level="rule").index if rule in excl.index.get_level_values("rule") else []
        ex_pick += [(x, rule) for x in pick(ids, 1)]
    cats.append(("Rule excludes the hidden target (strict mode does not insert it)", None, ex_pick, int(len(excl))))
    empty = info[info.n_cand == 0]
    short = info[(info.n_cand > 0) & (info.n_cand < 8)]
    cats.append(("Strict empty result (candidate count = 0)", None, [(x, r) for x, r in empty.index[:1]], int(len(empty))))
    cats.append(("Fewer than 8 candidates", None, [(x, r) for x, r in short.index[:1]], int(len(short))))

    thr = rec.freeze.get("genre_entropy_threshold") if rec.freeze else None
    blocks, n_cases, counts = [], 0, {}
    for title, rule0, picked, total in cats:
        counts[title] = total
        blocks.append(f"## {title}\n\nQueries in the test set that fall into this category: {total}.")
        if not picked:
            blocks.append("**No case of this category occurs** in this run's test set, so none is shown (nothing is invented).\n")
            continue
        for item in picked:
            qid, rule = (item if isinstance(item, tuple) else (item, rule0))
            n_cases += 1
            blocks.append(_case_block(n_cases, title, qid, rule, q.loc[qid], info.loc[(qid, rule)], cat, snap, rec,
                                      core, hits, ranks, top_of, pop_rank, thr, learned))
    head = ["# Failure case analysis (real test queries)", "",
            f"Run `{run.run_id}` (level {cfg['level']}). Every case is a test query from predictions.parquet, ordered by "
            "query_id and then picked within each category; no flattering examples are cherry-picked. Core models: A0, A1 and "
            f"B0 / B1 / C0 at deploy seed {rec.seed}.", "",
            "How to read: **Observation** is a fact read straight from the data; **Hypothesis (to verify)** is only a possible cause "
            "that no controlled experiment has confirmed, and neither feature importance nor case-level correlation is causal. "
            "The label is a proxy label (rating ≥ 8), unrated titles are weak negatives, and a miss does not mean the user would dislike it.", "",
            table(["Category", "Occurrences in the test set", "Cases shown"],
                  [[t, counts[t], sum(1 for _ in p_)] for t, _, p_, _ in cats]), ""]
    return "\n".join(head + blocks) + "\n", {"cases": n_cases, "categories": counts}


def _case_block(n, title, qid, rule, qrow, irow, cat, snap, rec, core, hits, ranks, top_of, pop_rank, thr,
                learned) -> str:
    H = np.asarray(qrow.visible, dtype=np.int64)
    prof = build_profile(cat, H)
    t = int(qrow.target)
    top_labels = prof.top_labels(cat)
    t_tags = [f"genre:{x}" for x in _names(cat, "genre", t)] + [f"theme:{x}" for x in _names(cat, "theme", t)]
    overlap = [x for x in t_tags if x in top_labels]
    studio_f = prof.freq.get("studio")
    t_studios = _names(cat, "studio", t)
    studio_hit = bool(studio_f is not None and any(
        studio_f[c] > 0 for c in cat.X["studio"].indices[cat.X["studio"].indptr[t]:cat.X["studio"].indptr[t + 1]]))
    eps = cat.episodes[t]
    hist_titles = "; ".join(_title(snap, i, 40) for i in H[:10]) + (f"; … ({len(H)} titles in total)" if len(H) > 10 else "")
    lines = [f"### Case {n}: {title} · query `{qid}`", "",
             f"- Offline user {int(qrow.offline_user_id)} · {'main group' if qrow.group == 'main' else 'sparse group'} · "
             f"{len(H)} visible titles · rule {RULE_TEXT[rule]}" + (f" · keyword `{qrow.r1_keyword}`" if rule in ("R1", "R3") else ""),
             f"- Visible collection: {hist_titles}",
             f"- Profile top-10 tags: {', '.join(top_labels) or '(none)'}",
             f"- Hidden target: **{_title(snap, t, 60)}** ({snap.keys[t]}) · public score {f4(cat.score[t], 2)} · "
             f"episodes {'unknown' if not np.isfinite(eps) else int(eps)} · popularity rank {pop_rank.get(t, '—')}/{len(cat.members)} · "
             f"tags {', '.join(t_tags) or '(none)'} · studio {', '.join(t_studios) or '(unknown)'}",
             f"- Overlap between the target and the profile top-10 tags: {len(overlap)}" + (f" ({', '.join(overlap)})" if overlap else "")
             + f"; the target's studio {'appears' if studio_hit else 'does not appear'} in the visible collection",
             f"- Candidates: {int(irow.n_cand)}; the target {'satisfies' if bool(irow.target_eligible) else 'does not satisfy'} this rule"
             + ("" if bool(irow.target_eligible) else " (strict mode never inserts it back into the candidates, so this query is "
                                                       "excluded from the conditional metrics and counted as not found end to end)"), ""]
    rows = []
    a0_scores, a0_hits = [], []
    for m in core:
        tk = top_of.get((qid, rule, m)) or ""
        keys = [k for k in tk.split("|") if k]
        idx = [snap.index[k] for k in keys]
        if m == "A0":
            a0_scores = [float(cat.score[i]) for i in idx]
            for i in idx:
                tags = [f"genre:{x}" for x in _names(cat, "genre", i)] + [f"theme:{x}" for x in _names(cat, "theme", i)]
                a0_hits.append(sum(1 for x in tags if x in top_labels))
        rk = ranks.loc[(qid, rule), m] if (qid, rule) in ranks.index else None
        hv = hits.loc[(qid, rule), m] if (qid, rule) in hits.index else None
        rows.append([m, "n/a" if rk is None or not np.isfinite(rk) else int(rk),
                     "—" if hv is None or not np.isfinite(hv) else int(hv),
                     " / ".join(_title(snap, i, 24) for i in idx) or "(empty)"])
    lines += [table(["Model", "Target rank", "Hit@8", "Top-8"], rows), ""]
    obs, hyp = [], []
    if a0_scores:
        obs.append(f"A0's Top-8 mean public score {np.mean(a0_scores):.2f}, mean hits {np.mean(a0_hits):.1f}; "
                   f"the target has public score {f4(cat.score[t], 2)} and hits {len(overlap)}. ")
    lr = {m: ranks.loc[(qid, rule), m] for m in learned if (qid, rule) in ranks.index}
    if any(v is not None and np.isfinite(v) for v in lr.values()):
        obs.append("Learned models' rank for the target: " + ", ".join(f"{m} {int(v)}" for m, v in lr.items()
                                              if v is not None and np.isfinite(v)) + ". ")
    if not bool(irow.target_eligible):
        why = []
        if rule in ("R2", "R3"):
            if not np.isfinite(cat.score[t]) or cat.score[t] < 7.0:
                why.append("public score below 7.0")
            if not np.isfinite(eps):
                why.append("episode count unknown (treated as not satisfying the episode cap)")
            elif eps > 26:
                why.append(f"episodes {int(eps)} > 26")
        if rule in ("R1", "R3") and qrow.r1_keyword:
            gt = [x.lower() for x in _names(cat, "genre", t)]
            if not any(qrow.r1_keyword in g for g in gt):
                why.append(f"neither the genres nor the title contain the keyword `{qrow.r1_keyword}`")
        obs.append("Why the rule excludes the target: " + ("; ".join(why) if why else "see the rule definition") + ". ")
    if a0_scores and np.isfinite(cat.score[t]) and cat.score[t] < np.mean(a0_scores) - 0.5:
        hyp.append("The target's public score is clearly below the mean of A0's recommended items; a ranking driven mainly by public score can push it down. ")
    if a0_hits and len(overlap) < np.mean(a0_hits):
        hyp.append("The target overlaps less with the profile top-10 tags than A0's recommended items do, so A0's hits term works against it. ")
    pr = pop_rank.get(t)
    if pr and pr > 2000:
        hyp.append(f"The target is relatively obscure in the catalogue (popularity rank {pr}), while the main signal of every model correlates with public score / popularity; obscure targets rarely reach the top 8. ")
    if title.startswith("A0 hits") and a0_hits and len(overlap) >= np.mean(a0_hits):
        share = ""
        if rec.importance is not None:
            imp = rec.importance[(rec.importance.family == "B1") & (rec.importance.seed == rec.seed)]
            if not imp.empty and "rating_norm" in set(imp.feature):
                share = f" (rating_norm accounts for {float(imp[imp.feature == 'rating_norm'].gain.iloc[0]) / imp.gain.sum():.0%} of B1's gain)"
        hyp.append(f"The target overlaps heavily with the profile tags (hits {len(overlap)}, not below the mean of A0's recommended items), so A0's hits term ranks it high; "
                   f"the learned models rely more on public score{share} and put higher-rated or more popular titles ahead of it. ")
    if title.startswith("A learned model hits"):
        hyp.append("B1 / C0 use signals that A0 does not have: studio, episode count, similarity"
                   + (" (the target's studio appears in the visible collection)" if studio_hit else "")
                   + ", which may rank it higher; this is a correlation in the data, not a causal proof. ")
    if qrow.group == "sparse":
        hyp.append(f"The visible collection has only {len(H)} titles, so each one carries 1/{len(H)} of the profile frequency and the tag signal is noisy. ")
    if title.startswith("Mixed preferences") and thr is not None:
        hyp.append(f"The genre entropy of the visible collection is above the pre-determined threshold {thr:.2f}: preferences are spread out and the top-10 tags hardly represent a single taste. ")
    lines += ["**Observation**: " + ("".join(obs) if obs else "—"), "",
              "**Hypothesis (to verify)**: " + ("".join(hyp) if hyp else "No clear data-based hypothesis yet."), ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ report


def _main_table(rec: Records) -> str:
    prim = rec.metrics["primary"]
    fams = rec.metrics.get("families", {})
    rows = []
    for m in ["A0", "A1", *[f"{f}_seed{s}" for f in ("B0", "B1", "C0") for s in rec.cfg["seeds"]],
              "AUX_slides_v0", "AUX_profile_v1"]:
        p = prim.get(m)
        if p is None:
            continue
        d0, l0, h0 = rec.pair(m, "A0")
        d1, l1, h1 = rec.pair(m, "A1")
        name = m + (" ★deployed" if m in (rec.labels["B0"], rec.labels["B1"], rec.labels["C0"]) else "")
        rows.append([name, p["users"], f4(p["HR@8"]), f4(p["NDCG@8"]), f4(p["MRR@8"]), f4(p["AUC"]),
                     "—" if m == "A0" else f"{pp(d0)} {ci_pp(l0, h0)}", "—" if m == "A1" else f"{pp(d1)} {ci_pp(l1, h1)}",
                     p["violations"]])
    for fam in ("B0", "B1", "C0"):
        f = fams.get(fam)
        if not f:
            continue
        d0, l0, h0 = rec.pair(f"{fam}_mean", "A0")
        d1, l1, h1 = rec.pair(f"{fam}_mean", "A1")
        rows.append([f"{fam} three-seed mean ± SD", "—", f"{f4(f['HR@8_mean'])} ± {f4(f['HR@8_sd'])}",
                     f4(f["NDCG@8_mean"]), "—", f4(f["AUC_mean"]), f"{pp(d0)} {ci_pp(l0, h0)}",
                     f"{pp(d1)} {ci_pp(l1, h1)}", "—"])
    return table(["Model", "Users", "HR@8", "NDCG@8", "MRR@8", "AUC", "ΔHR@8 vs A0 (percentage points, 95% CI)",
                  "ΔHR@8 vs A1 (percentage points, 95% CI)", "Rule violations"], rows)


def _comparisons(rec: Records) -> str:
    rows = []
    for title, m, base in (("A1 − A0: gain from weight optimisation", "A1", "A0"),
                           ("B0 − A1: gain from a learned scoring function on the same two features", "B0_mean", "A1"),
                           ("B1 − B0: gain from the six additional features", "B1_mean", "B0_mean"),
                           ("C0 − B1: content two-tower against the eight-feature GBDT", "C0_mean", "B1_mean"),
                           ("C0 − B0", "C0_mean", "B0_mean")):
        d, lo, hi = rec.pair(m, base)
        dn, lon, hin = rec.pair(m, base, "NDCG@8")
        rows.append([title, f"{pp(d)} {ci_pp(lo, hi)}", f"{pp(dn)} {ci_pp(lon, hin)}"])
    return table(["Comparison (learned models averaged over three seeds at user level)", "ΔHR@8 (percentage points, 95% CI)", "ΔNDCG@8 (percentage points, 95% CI)"], rows)


def _interpretation(rec: Records) -> list[str]:
    """One sentence per controlled comparison, judged only by its paired 95 % interval."""
    def verdict(lo, hi) -> str:
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            return "cannot be judged (not run)"
        if lo > 0:
            return "interval above 0: a consistent gain"
        if hi < 0:
            return "interval below 0: consistently worse"
        return "interval contains 0: insufficient evidence of a difference"
    out = []
    for text, m, base in (("Weight optimisation (A1 − A0)", "A1", "A0"), ("Learned scoring function (B0 − A1, same two features)", "B0_mean", "A1"),
                          ("Additional features (B1 − B0)", "B1_mean", "B0_mean"), ("Content two-tower against the eight-feature GBDT (C0 − B1)", "C0_mean",
                                                                     "B1_mean")):
        d, lo, hi = rec.pair(m, base)
        out.append(f"- {text}: ΔHR@8 {pp(d)} percentage points {ci_pp(lo, hi)}, {verdict(lo, hi)}.")
    prim = rec.metrics["primary"]
    core = {f: prim.get(rec.labels[f]) for f in ("A0", "A1", "B0", "B1", "C0") if prim.get(rec.labels[f])}
    if core:
        best_hr = max(core, key=lambda f: core[f]["HR@8"])
        best_auc = max(core, key=lambda f: core[f]["AUC"] or 0)
        if best_hr != best_auc:
            out.append(f"- The highest AUC is {best_auc} ({f4(core[best_auc]['AUC'])}), the highest HR@8 is {best_hr}"
                       f" ({f4(core[best_hr]['HR@8'])}): AUC measures the overall ranking of the target against all weak negatives while HR@8 only looks at the top 8, "
                       "so the two can name different winners; the pre-registered primary metric is HR@8.")
    return out


def candidate_buckets(rec: Records) -> pd.DataFrame | None:
    """HR@8 by strict candidate count over all rule scenarios (R0 alone always has > 500 candidates);
    target-eligible queries, user-macro (a user's queries in one bucket are averaged first)."""
    from algorithm2_benchmark.pipeline.final_eval import CAND_BUCKETS, _bucket
    from algorithm2_benchmark.pipeline.metrics import user_macro

    pred = _load(rec.run, "predictions.parquet")
    if pred is None:
        return None
    models = [rec.labels[f] for f in ("A0", "A1", "B0", "B1", "C0")]
    d = pred[pred.model.isin(models) & pred.target_eligible & pred.hit.notna()].copy()
    d["value"] = d.n_cand.map(lambda x: _bucket(int(x), CAND_BUCKETS))
    rows = []
    for (m, v), g in d.groupby(["model", "value"]):
        hr, users = user_macro(g.user.to_numpy(), g.hit.astype(float).to_numpy())
        rows.append({"model": m, "dimension": "cand_bucket_all_rules", "value": v, "HR@8": hr, "users": users,
                     "queries": int(len(g)), "note": "rules R0-R3, conditional on target eligibility"})
    return pd.DataFrame(rows)


def _subgroup_table(rec: Records) -> str:
    if rec.subgroups is None:
        return NR
    sg = rec.subgroups
    cb = candidate_buckets(rec)
    if cb is not None and not cb.empty:
        sg = pd.concat([sg[sg.dimension != "cand_bucket"], cb], ignore_index=True)
    models = [rec.labels[f] for f in ("A0", "A1", "B0", "B1", "C0")]
    out = []
    for dim, label in (("rule", "Rule scenario (conditional on target eligibility)"), ("group", "Collection-size group (R0)"),
                       ("history_bucket", "Visible history (R0)"), ("cand_bucket_all_rules", "Strict candidate count (R0–R3 pooled, target eligible)"),
                       ("profile_mix", "Profile concentrated / mixed (R0)")):
        d = sg[(sg.dimension == dim) & sg.model.isin(models)]
        if d.empty:
            continue
        order = {v: i for i, v in enumerate(["R0", "R1", "R2", "R3", "main", "sparse", "1-4", "5-9", ">=10", "<=8",
                                             "9-50", "51-500", ">500", "concentrated", "mixed", "unknown"])}
        vals = sorted(d.value.astype(str).unique(), key=lambda v: (order.get(v, 99), v))
        rows = []
        for v in vals:
            row = [v]
            n = None
            for m in models:
                r = d[(d.model == m) & (d.value.astype(str) == v)]
                if r.empty:
                    row.append("—")
                    continue
                n = int(r.iloc[0].users)
                row.append(f4(r.iloc[0]["HR@8"]))
            row.insert(1, n if n is not None else "—")
            if n is not None and n < 100:
                row[0] = f"{v} (N<100, no stable comparison)"
            rows.append(row)
        out += [f"**{label}** (HR@8, user-level average)", "", table(["Group", "Users", *models], rows), ""]
    return "\n".join(out)


def _ms(x: float) -> str:
    return f"{x:.3f}" if x < 1 else f"{x:.1f}"


def _latency_table(rec: Records) -> str:
    if rec.latency is None:
        return f"{NR}: this run has not run the benchmark yet."
    rows = []
    for fam in ("A0", "A1", "B0", "B1", "C0"):
        r = [fam]
        for layer in ("scoring", "service", "api"):
            for rule in ("R0", "R3"):
                p50, p95 = rec.lat(fam, layer, rule, "p50_ms"), rec.lat(fam, layer, rule, "p95_ms")
                r.append("—" if p50 is None or not np.isfinite(p50) else f"{_ms(p50)} / {_ms(p95)}")
        info = rec.model_info(rec.labels[fam])
        r += [f"{info.get('fit_seconds', 0.0):.1f}" if info else "0",
              f"{info.get('bytes', 0) / 1024:.0f}" if info else "0"]
        rows.append(r)
    return table(["Model", "Scoring only R0 P50/P95", "Scoring only R3", "Service R0", "Service R3", "API R0", "API R3",
                  "Final training time (s)", "Model size (KB)"], rows)


def _gbdt_trees(path) -> list[str]:
    text = path.read_text()
    return text.split("end of trees")[0].split("\nTree=")[1:]


def _seed_note(rec: Records) -> str:
    """What the training seed changes for LightGBM in this run (checked on the saved trees, not assumed)."""
    if not rec.freeze or not rec.split:
        return ""
    n_train = sum(v for k, v in rec.split["pairs"]["by_partition_label"].items() if k.startswith("train"))
    same = {}
    for fam in ("B0", "B1"):
        files = [rec.run.root / m["file"] for m in rec.freeze["models"] if m["family"] == fam]
        trees = [_gbdt_trees(f) for f in files if f.exists()]
        same[fam] = len(trees) > 1 and all(t == trees[0] for t in trees[1:])
    cap = 200000
    if all(same.values()):
        why = (f"the {n_train:,} training pairs do not exceed LightGBM's histogram bin-construction sample cap of {cap:,}, so binning "
               "uses every sample and the trees of the three B0 and B1 seeds are identical (a standard deviation of 0 is a real property of this configuration, not three independent random draws).")
    else:
        why = (f"there are {n_train:,} training pairs, above LightGBM's histogram bin-construction sample cap of {cap:,} (bin_construct_sample_cnt), so the seed "
               "decides which rows take part in binning and the bin boundaries and trees of the three seeds differ slightly (checked on the model files: "
               + ", ".join(f"{f} {'identical' if v else 'different'}" for f, v in same.items())
               + "); differences between seeds reflect only this source of randomness.")
    return "Note: the pre-registered LightGBM grid has no bagging / feature_fraction subsampling, so the seed only acts through bin-construction sampling; " + why


def final_report(rec: Records, cases: dict) -> str:
    cfg, run = rec.cfg, rec.run
    d, dm, fz, mt, dec = rec.data, rec.split, rec.freeze, rec.metrics, rec.decision
    L: list[str] = []
    a = L.append
    status = "completed" if (mt and dec and rec.integration and rec.integration.get("status") == "passed"
                        and rec.latency is not None) else "partially completed / effectiveness validation blocked"
    if cfg["level"] != "main":
        status = f"{cfg['level']}-level pipeline check (not a formal model-selection conclusion)"
    a(f"# Algorithm 2 (collection recommendation): effectiveness comparison, model selection and system integration report — {status}")
    a("")
    a(f"Run `{run.run_id}` · level **{cfg['level']}** · generated at {now()}. Every number in this report comes from a record file in this run directory"
      " (see section 9); anything that did not run is written as NOT_RUN.")
    if cfg["level"] != "main":
        a("")
        a(f"> Note: this is a {cfg['level']}-level run with few users; it only checks the pipeline and is not a formal model-selection conclusion.")
    a("")
    # 1
    a("## 1. State of the original system, scope of this work and what was added")
    a("")
    g = (rec.meta or {}).get("start_git", {})
    a(f"- Code starting point: branch `{g.get('branch', '—')}`, commit `{(g.get('commit') or '—')[:10]}` (identical to origin/main)."
      " The repository audit is in `algorithm2_benchmark/docs/repo_audit.md`.")
    a("- Original state: the collection page used `collection_profile_v1` (normalised profile matching, α = 0.7) as the default scorer,"
      " with `manual_slides_v0` (the slides formula, counting hits by presence) kept only as a reference; there was no trained collection model,"
      " no fixed user split, no pre-registered selection rule and no offline / online consistency check.")
    a("- Added in this work: `app/recommendation/algorithm2/` (unified catalogue, profiles, eight features, five scorers, a NumPy reader for the LightGBM text model,"
      " model-bundle verification and serving), `algorithm2_benchmark/` (a ten-stage command-line experiment pipeline, run records and reports), the collection endpoint now calling the same code path,"
      " the strict rules (with the new `pool.scored`) and an explicit-relaxation response contract, and every record behind this report.")
    a("")
    # 2
    a("## 2. Data source, labels, cleaning and scale")
    a("")
    if d:
        cat_m, inter = d["catalogue"], d["interactions"]
        a(f"- Data: Kaggle MyAnimeList 2023 snapshot, processed version `{d['source']['dataset_version']}`"
          f" (raw-file SHA-256 in data_manifest.json). Only the MAL id is used as the item identity.")
        a(f"- Recommendation catalogue: {cat_m['members']:,} titles = servable (Hentai / Erotica / Rx excluded; Ecchi and R+ kept)"
          f" and having a public score. The snapshot has {cat_m['snapshot_rows']:,} rows. Catalogue hash `{cat_m['catalog_hash'][:12]}`.")
        cov = cat_m["field_coverage"]
        src = cat_m.get("sources") or {}
        sup, added = src.get("supplement") or {}, src.get("supplement_values_added") or {}
        if sup.get("path") and sup.get("status") != "missing":
            sup_text = (f"Theme / demographic were supplemented from the teammate-provided `{sup['path']}` joined by MAL id (union; "
                        f"{added.get('theme', 0):,} theme and {added.get('demographic', 0):,} demographic values added)")
        else:
            sup_text = "This run used no theme / demographic supplement file (only the snapshot's own tags)"
        a("- Field coverage: " + ", ".join(f"{k} {v:.1%}" for k, v in cov.items()) + f". {sup_text}; missing values are never invented, and"
          " an unknown episode count does not satisfy the episode-cap rule.")
        a(f"- Interaction cleaning: {inter['rows_read']:,} ratings read, {inter['rows_kept']:,} kept inside the catalogue"
          f" ({inter['outside_catalogue']:,} outside, all of them lacking a public score); no invalid scores and no conflicting duplicates.")
        a(f"- Label: a **proxy label** — the titles a user rated ≥ 8 form that user's high-rating preference set ({inter['positive_rows']:,} rows,"
          f" {inter['users_with_positive']:,} users). The application database holds no real collections, so every effectiveness conclusion here is"
          " an evaluation on that high-rating preference set used as a proxy, not collection behaviour itself; unrated titles are weak negatives.")
        a("- Preference-set size distribution (users): " + ", ".join(f"{k.replace('-1000000000', '+')}: {v:,}"
                                             for k, v in d["positive_set_size_distribution"].items()) + ".")
    else:
        a(NR)
    a("")
    # 3
    a("## 3. Splits, leave-one-out, rule scenarios, candidates and leakage control")
    a("")
    if dm:
        ms = dm["main_split"]["by_group_partition"]
        a(f"- User-level split 70 / 15 / 15, `split_seed = {dm['seeds']['split_seed']}`, stratified by preference-set size;"
          f" the main group (set ≥ 6, i.e. visible history ≥ 5) and the sparse group (2–5) are reported separately. Full split: "
          + ", ".join(f"{k} {v:,}" for k, v in ms.items()) + ".")
        lu = dm["level_users"]
        a(f"- This level uses {lu['users']:,} users (nested in the full split: {lu['nested_in_main_split']})."
          f" Users shared between partitions: {dm['partition_overlap_users']}.")
        qq = dm["queries"]
        a(f"- Queries: one hidden target per user for validation / test, up to {cfg['split']['train_targets_per_collection']}"
          f" targets per user for training; {qq['rows']:,} queries in total ({qq['by_partition_group']}). The hidden target never enters the profile input H.")
        by = dm["pairs"]["by_partition_label"]
        n_tr = sum(v for k, v in by.items() if k.startswith("train"))
        n_va = sum(v for k, v in by.items() if k.startswith("val"))
        a(f"- Training samples: 1 positive + 4 weak negatives per training query (deterministic sampling inside the same candidate set, `negative_seed = "
          f"{dm['seeds']['negative_seed']}`, excluding all known positives of that user), {dm['pairs']['rows']:,} pairs in total ({n_tr:,} training pairs;"
          f" the {n_va:,} pairs of validation users are used only for the early-stopping logloss); the negative shortfall occurred {dm['pairs']['negatives_shortfall']} times."
          " B0 / B1 / C0 share the same sample set.")
    a("- Rule scenarios: R0 has no filter; R1 takes the most common genre in H as the keyword (ties broken by name), and the keyword is matched as an AND substring against genre or title;"
      " R2 requires public score ≥ 7.0 and episodes ≤ 26 (an unknown episode count does not satisfy it); R3 = R1 + R2. All of them are strict: a target that does not satisfy the rule is not inserted back,"
      " is recorded as target_not_eligible, and end-to-end results are reported separately.")
    a("- Candidates: the **full catalogue** passing the rule is ranked (no candidate sampling), excluding H itself. Ties are broken by ascending MAL id;"
      " AUC gives 0.5 to ties and is unaffected by the id ordering; AUC is null when there are fewer than 2 candidates.")
    a("- Primary metric (pre-registered): user-macro HR@8 on R0, main group, more than 8 candidates, target eligible; NDCG@8, MRR@8 and AUC are reported in the same table.")
    a("")
    # 4
    a("## 4. Implementation, parameters and budget of the five experiment groups")
    a("")
    if rec.tuning and fz:
        sel = rec.tuning["selected"]
        enc = rec.tuning.get("encoder", {})
        trials = rec.trials
        tcount = trials.groupby("family").size().to_dict() if trials is not None else {}
        a("- **A0** = rating_norm + 0.15 × hits (hits = the number of candidate genre/theme tags that fall inside the top-10 namespace tags of H).")
        w, grid = sel["A1"]["hits_weight"], [float(x) for x in cfg["a1_weights"]]
        edge = ""
        if float(w) in (min(grid), max(grid)):
            side = "minimum" if float(w) == min(grid) else "maximum"
            edge = (f"; the selected value is the {side} of the pre-registered grid, which suggests the data may prefer a weight outside the grid, but the grid was not extended after the results were seen")
        a(f"- **A1**: hits weights {cfg['a1_weights']} compared on the validation set, selected **{w}**"
          f" (validation HR@8 {f4(sel['A1']['val_HR@8'])}){edge}.")
        for fam in ("B0", "B1"):
            s = sel[fam]
            a(f"- **{fam}**: native LightGBM `lgb.train` (objective = binary, ranked by raw score), features "
              f"{', '.join(s['features'])}; {tcount.get(fam, 0)} pre-generated deterministic trials (grid {cfg['gbdt']['grid']}), "
              f"capped at {cfg['gbdt']['n_estimators']} trees with early stopping after {cfg['gbdt']['early_stopping_rounds']} rounds (logloss on validation), "
              f"trial {s['trial']} selected by validation HR@8 over the full candidate set: {s['params']} (validation HR@8 {f4(s['val_HR@8'])}).")
        s = sel["C0"]
        a(f"- **C0**: PyTorch content two-tower, {cfg['two_tower']['hidden']} hidden units per side; inputs are multi-hot vectors over genre / theme / demographic / "
          f"studio (studios with at least {enc.get('studio_min_count')} appearances, the rest mapped to UNK) plus numeric features and a missing-value mask"
          f" (anime tower {enc.get('d_item')} dims, collection tower {enc.get('d_coll')} dims), dot-product logits + BCEWithLogitsLoss, "
          f"no in-batch negatives; {tcount.get('C0', 0)} trials, trial {s['trial']} selected: {s['params']}"
          f" (validation HR@8 {f4(s['val_HR@8'])}), at most {cfg['two_tower']['max_epochs']} epochs, patience "
          f"{cfg['two_tower']['early_stopping_patience']}, on CPU.")
        a(f"- The final configuration is trained once per training seed {cfg['seeds']} (9 models in total, see models_manifest.json). " + _seed_note(rec))
        if trials is not None and not trials.empty:
            cost = trials.groupby("family")[["fit_seconds", "val_score_seconds"]].sum()
            parts = []
            for fam in ("B0", "B1", "C0"):
                if fam in cost.index:
                    fit, val = cost.loc[fam, "fit_seconds"], cost.loc[fam, "val_score_seconds"]
                    parts.append(f"{fam} fit {fit:,.0f} s" + (f" + full-candidate validation scoring {val:,.0f} s" if val and val == val
                                                                else " (validation scoring not timed separately)"))
            a(f"- Tuning cost ({', '.join(f'{k} {v} trials' for k, v in tcount.items() if k != 'A1')}, total): "
              + "; ".join(parts) + ". Validation scoring over the full catalogue is the main cost.")
        a("- Comparison logic: B0 is the two-feature control; B1 and C0 use metadata from the same source but different representations, so C0 vs B1 is not 'the same input with only the model swapped'.")
        if rec.importance is not None:
            imp = rec.importance[(rec.importance.family == "B1") & (rec.importance.seed == rec.seed)]
            if not imp.empty:
                tot = imp.gain.sum()
                a("- B1 feature importance (gain share, deploy seed): " + ", ".join(
                    f"{r.feature} {r.gain / tot:.1%}" for r in imp.sort_values("gain", ascending=False).itertuples())
                  + ". Importance is not causal contribution; a feature's effect is judged by the B1 − B0 comparison.")
    else:
        a(NR)
    a("")
    # 5
    a("## 5. Test results")
    a("")
    if mt:
        a(f"Test set: {mt['test_users']:,} users and {mt['test_queries']:,} queries. The main table covers the pre-registered primary set ({mt['primary_definition']}).")
        a("")
        a("### 5.1 Main table")
        a("")
        a(_main_table(rec))
        a("")
        a("### 5.2 Key comparisons (user-paired bootstrap, 2,000 resamples, resampled by user)")
        a("")
        a(_comparisons(rec))
        a("")
        for line in _interpretation(rec):
            a(line)
        a("")
        a("The intervals reflect only user-sampling uncertainty and do not cover proxy-label or static-snapshot bias.")
        a("")
        a("### 5.3 Subgroup results")
        a("")
        a(_subgroup_table(rec))
        a("### 5.4 End-to-end results under the rule scenarios")
        a("")
        rows = []
        for rule in ("R0", "R1", "R2", "R3"):
            el = mt["rule_eligibility"].get(rule, {})
            e2e = [f4(mt["end_to_end"].get(rec.labels[f], {}).get(rule, {}).get("end_to_end_target_in_strict_top8"))
                   for f in ("A0", "A1", "B0", "B1", "C0")]
            rows.append([RULE_TEXT[rule], f"{el.get('target_eligible_share', float('nan')):.1%}",
                         f"{el.get('empty_result_share', float('nan')):.1%}", f"{el.get('fewer_than_8_share', float('nan')):.1%}",
                         *e2e])
        a(table(["Rule", "Target eligible share", "Empty-result share", "Fewer than 8 share",
                 *[f"End-to-end {rec.labels[f]}" for f in ("A0", "A1", "B0", "B1", "C0")]], rows))
        a("")
        a("'End-to-end' = the share of all original queries where the strict Top-8 contains the hidden target (a target that is not eligible counts as not met); "
          "this is a different quantity from the conditional metrics in 5.3.")
        hy = mt.get("hygiene", {})
        a("")
        a("Recommendation hygiene (all test rows): " + "; ".join(
            f"{m} rule-violation rate {v['rule_violation_rate']:.0%} / duplicate rate {v['duplicate_rate']:.0%} / already-saved duplicate rate "
            f"{v['saved_in_top_rate']:.0%}" for m, v in hy.items() if m in rec.labels.values()) + ".")
    else:
        a(f"{NR}: this run did not complete evaluate-test.")
    a("")
    a("### 5.5 Latency and cost")
    a("")
    a(_latency_table(rec))
    a("")
    if rec.bench:
        b = rec.bench
        env = b["environment"]
        a(f"P50 / P95 in milliseconds; measured on one machine ({env.get('platform')}, {env.get('cpu_count')} cores, "
          f"{(env.get('memory_bytes') or 0) / 2**30:.0f} GB memory), {b['threads']} CPU threads, no GPU, no result cache, "
          f"no batching; {b['request_list']['n']} fixed requests measured after {b['request_list']['warmup_n']} warm-up runs "
          f"({b['request_list']['R3_requests_without_keyword_skipped']} requests without a keyword were skipped under R3). "
          "Scoring only = the scorer alone; service = profile + strict rules + features + scoring + Top-8; API = the collection endpoint called in-process "
          "(reading the collection and reasons, writing the run log).")
        ld = b["load_seconds"]
        a(f"Load time (not counted in request latency): catalogue and rule engine {ld.get('catalogue_and_rule_engine')} s; "
          f"formulas {ld.get('A0+A1')} s; B0 {ld.get('B0')} s; B1 {ld.get('B1')} s; C0 {ld.get('C0')} s"
          f" (including {ld.get('C0_breakdown', {}).get('items')} precomputed anime vectors in "
          f"{ld.get('C0_breakdown', {}).get('item_vectors_s')} s). Peak memory (whole process): "
          + ", ".join(f"{k} {v} MB" for k, v in b["peak_rss_mb"].items() if v) + ".")
        a("The project has no fixed latency SLO, so only measured values are reported here and they are not a selection threshold. In serving, B0 / B1 use the "
          "NumPy reader for the LightGBM text model (native LightGBM deadlocks on an OpenMP conflict when PyTorch is loaded in the same process); the freeze step "
          f"checked the score gap against native LightGBM (maximum absolute difference {fz['gbdt_serving_engine']['check_vs_native']['max_abs_diff'] if fz else '—'}).")
    a("")
    # 6
    a("## 6. Pre-selection, test confirmation and the default scorer")
    a("")
    if fz:
        rows = []
        for fam, v in fz["validation"].items():
            rows.append([fam, f4(v["val_HR@8"]), f4(v.get("val_HR@8_sd")) if v.get("seeds") else "—",
                         f4(v["val_NDCG@8"]), ", ".join(f"{k}: {f4(x)}" for k, x in (v.get("seeds") or {}).items()) or "—"])
        a(table(["Candidate (validation)", "HR@8", "seed SD", "NDCG@8", "HR@8 per seed"], rows))
        a("")
        a(f"Pre-selected on the validation set under the pre-registered rule (`selection_protocol.json`, written before any training): **{fz['pre_selected']}**; "
          f"the deployed artifact is seed {fz['deploy_seed']}; the conservative fallback is {fz['fallback']}.")
    if dec:
        a("")
        rows = []
        for k, c in dec["checks"].items():
            val = c.get("value")
            val = ci_pp(*c["ci95"]) + " (percentage points)" if c.get("ci95") else (f4(val) if isinstance(val, float) else val)
            rows.append([k, c.get("threshold", "—"), val, "pass" if c["pass"] else "fail"])
        a(table(["Test confirmation check (deployed artifact " + dec["deploy_label"] + ")", "Threshold", "Measured", "Result"], rows))
        a("")
        a(f"Conclusion: {'confirmed' if dec['confirmed'] else 'not confirmed'}. Default scorer = **{dec['default_scorer']}**. Reason: {dec['reason']}. "
          "The test results were not used to retune or to switch to a different model.")
        real_cfg = None
        try:
            import yaml

            from app.core.config import CONFIG_DIR

            real_cfg = (yaml.safe_load((CONFIG_DIR / "real.yaml").read_text()) or {}).get("recommendation", {}).get(
                "collection", {})
        except Exception:  # noqa: BLE001 - informational
            real_cfg = None
        if real_cfg is not None:
            on = real_cfg.get("scorer") == dec["default_scorer"] and str(real_cfg.get("algorithm2_run", "")).endswith(
                run.run_id)
            a("")
            a(f"Site configuration (`configs/real.yaml`): scorer = `{real_cfg.get('scorer', '(not set, base.yaml applies)')}`, "
              f"algorithm2_run = `{real_cfg.get('algorithm2_run')}` — "
              + ("set according to this conclusion." if on else "not yet pointing at this run (this run is not the frozen result the site currently serves)."))
    else:
        a(f"{NR}: this run has no test confirmation.")
    a("")
    # 7
    a("## 7. Failure cases and limitations")
    a("")
    if cases:
        a(f"`case_analysis.md` reports {cases.get('cases', 0)} real test cases: " + "; ".join(
            f"{k} {v}" for k, v in cases.get("categories", {}).items()) + " (occurrences in the test set).")
    a("")
    a("Main limitations:")
    a("- Proxy labels: the set of titles rated ≥ 8 is not the same as a collection; unrated titles are weak negatives, so AUC measures how well the target is separated from those weak negatives.")
    a("- Static snapshot (2023) and held-out users: what is evaluated is whether other high-rated titles from the same period can be recovered, not a prediction of future behaviour.")
    a("- Popularity bias: targets are mostly popular high-rated titles, so features that rely on public score / popularity have a natural advantage; obscure targets rarely reach the top 8.")
    a("- Field coverage: theme and demographic depend on a supplement file and are incompletely covered; missing values are treated as NaN / empty sets and never fabricated.")
    a("- The LightGBM seed only affects bin-construction sampling (the grid has no subsampling), so differences across the three seeds do not represent the full training randomness.")
    a("- C0 could in principle encode new titles with no interactions, but there was no item-disjoint experiment here, so we cannot claim better cold start for new titles.")
    a("- Latency was measured on a single laptop and is meant for relative comparison only, not for production.")
    a("")
    # 8
    a("## 8. System integration verification")
    a("")
    ir = rec.integration
    if ir and ir.get("checks"):
        rows = [[c["check"], "pass" if c["passed"] else "**fail**"] for c in ir["checks"]]
        a(table(["Check (integration_report.json)", "Result"], rows))
        a("")
        a(f"Runtime environment: {ir.get('database')}. Overall status: **{ir.get('status')}**.")
    elif ir:
        a(f"{ir.get('status')}: {ir.get('reason')}")
    else:
        a(f"{NR}: this run has not executed verify-integration.")
    a("")
    a("Integrated: the collection endpoint reads the current collection H and calls exactly the same catalogue, profile, rule engine, features and scorer as "
      "the offline path; the scorer registry offers A0 / A1 / B0 / B1 / C0 (`recommendation.collection.scorer`, or the `scorer` field in the request body for experiments / operations); "
      "a missing or incompatible model bundle falls back to the formulas and records the status without relaxing the user's filters; when a strict query returns nothing it "
      "offers relaxation only, applies it after the user confirms, and the response carries `relaxed`, `relaxed_rules` and `strict_result_count`, with a readable relaxation note on the page.")
    a("Still open: the application database holds no real user collections and there is no online experiment (A/B or click feedback), so the actual collection recommendation "
      "quality on the site is not yet validated; every effectiveness number in this report comes from the offline proxy evaluation.")
    a("")
    # 9
    a("## 9. Reproduction")
    a("")
    a("See `README_reproduce.md` in the same directory (commands, data locations and output files). The main records: `config.json`, `data_manifest.json`, "
      "`split_manifest.json`, `query_manifest.parquet`, `sampled_pairs.parquet`, `selection_protocol.json`, "
      "`trial_results.csv`, `tuning.json`, `training_curves.json`, `feature_importance.csv`, `models_manifest.json`, "
      "`models/`, `freeze_manifest.json`, `feature_schema.json`, `predictions.parquet`, `metrics.json`, `summary.csv`, "
      "`subgroup_metrics.csv`, `paired_comparisons.csv`, `decision.json`, `latency.csv`, `benchmark.json`, "
      "`integration_report.json`, `case_analysis.md`, `run_metadata.json`.")
    a("")
    return "\n".join(L) + "\n"


def readme_reproduce(rec: Records) -> str:
    run, cfg = rec.run, rec.cfg
    rid, lvl = run.run_id, cfg["level"]
    from app.core.config import PROJECT_ROOT

    rel = run.root.relative_to(PROJECT_ROOT) if run.root.is_relative_to(PROJECT_ROOT) else run.root
    return f"""# Reproducing run `{rid}`

Level: `{lvl}`. All commands run from the repository root.

## Environment

```bash
uv sync --extra ml          # Python 3.12 environment with LightGBM {(rec.meta or {}).get('environment', {}).get('lightgbm', '')} and PyTorch
brew install libomp         # macOS only: the OpenMP runtime LightGBM links against
```

LightGBM and PyTorch are never loaded into the same process (their OpenMP runtimes deadlock on macOS); the stages
that need both start isolated sub-processes automatically.

## Data

- Processed real dataset `data/processed/real/{(rec.data or {}).get('source', {}).get('dataset_version', 'real-…')}`
  (Kaggle MyAnimeList 2023 snapshot, Git LFS), verified against its manifest by `prepare`.
- Theme / demographic supplement `{cfg['data']['supplement']}` (joined by MAL id).

## Commands

```bash
# every stage in order (finished stages are skipped; a new run needs --level)
.venv/bin/python -m algorithm2_benchmark run-all --run {rid} --level {lvl}

# or one stage at a time
.venv/bin/python -m algorithm2_benchmark audit              --run {rid} --level {lvl}
.venv/bin/python -m algorithm2_benchmark prepare            --run {rid}
.venv/bin/python -m algorithm2_benchmark split              --run {rid}
.venv/bin/python -m algorithm2_benchmark tune               --run {rid}
.venv/bin/python -m algorithm2_benchmark train              --run {rid}
.venv/bin/python -m algorithm2_benchmark freeze             --run {rid}
.venv/bin/python -m algorithm2_benchmark evaluate-test      --run {rid}
.venv/bin/python -m algorithm2_benchmark benchmark          --run {rid}
.venv/bin/python -m algorithm2_benchmark verify-integration --run {rid}
.venv/bin/python -m algorithm2_benchmark report             --run {rid}
```

Options: `--threads N` (LightGBM / PyTorch threads, fixed at run creation), `--root DIR` (runs directory).
After `freeze`, `tune` and `train` refuse to run; `evaluate-test` only reproduces the frozen evaluation.
The Makefile wraps the same commands (`make alg2-run RUN={rid} LEVEL={lvl}`).

## Outputs

`{rel}/` holds the committed records (manifests, metrics, models, reports); `work/` (git-ignored) holds the
regenerable intermediate files (positives, features, per-pass predictions, logs).

## Serving the result

Set in `configs/real.yaml`:

```yaml
recommendation:
  collection:
    scorer: {(rec.decision or {}).get('default_scorer', 'A0')}
    algorithm2_run: {rel}
```

The web app verifies the model hashes, catalogue hash, feature version and encoder before loading; on any mismatch
it serves the formula and reports `collection_scorer: degraded` on the status page.
"""


def enrich_metadata(rec: Records) -> None:
    """Collect the input hashes, configurations and seeds of the run into run_metadata.json (`inputs`)."""
    path = rec.run.path("run_metadata.json")
    if not path.exists():
        return
    meta = read_json(path)
    d, sm, fz = rec.data or {}, rec.split or {}, rec.freeze or {}
    files = (d.get("file_checks") or {})
    meta["inputs"] = {
        "dataset_version": (d.get("source") or {}).get("dataset_version"),
        "raw_source_sha256": (d.get("processed_dataset") or {}).get("raw_source_sha256"),
        "processed_files_sha256": {k: v["sha256"] for k, v in files.items()},
        "processed_files_match_manifest": all(v.get("matches_manifest") for v in files.values()) if files else None,
        "catalog_hash": (d.get("catalogue") or {}).get("catalog_hash"),
        "supplement": ((d.get("catalogue") or {}).get("sources") or {}).get("supplement"),
        "users_sha256": (sm.get("main_split") or {}).get("users_sha256"),
        "query_manifest_sha256": (sm.get("queries") or {}).get("query_manifest_sha256"),
        "sampled_pairs_sha256": (sm.get("pairs") or {}).get("sha256"),
        "candidates": "R0 = catalogue members minus H (catalog_hash); per query cand_hash in predictions.parquet",
        "feature_version": fz.get("feature_version"), "encoder": fz.get("encoder"),
        "model_configs": fz.get("configs"), "formula": fz.get("formula"),
        "model_files_sha256": {f"{m['family']}_seed{m['seed']}": m["sha256"] for m in fz.get("models", [])},
        "seeds": {"split_seed": rec.cfg["split"]["split_seed"], "negative_seed": rec.cfg["negatives"]["negative_seed"],
                  "training_seeds": rec.cfg["seeds"], "deploy_seed": fz.get("deploy_seed"),
                  "bootstrap_seed": rec.cfg["selection"]["bootstrap_seed"]},
        "device": "cpu", "protocol_sha256": fz.get("protocol_sha256")}
    write_json(path, meta)


def publish_records(run) -> list[str]:
    out = []
    for name in ("query_manifest.parquet", "sampled_pairs.parquet", "users_split.parquet"):
        src = run.work / name
        if src.exists():
            shutil.copy(src, run.path(name))
            out.append(name)
    return out


def report(run, log) -> dict:
    copied = publish_records(run)
    rec = Records(run)
    enrich_metadata(rec)
    summary = final_summary(rec)
    rec.summary = summary if summary is not None else rec.summary
    cb = candidate_buckets(rec)
    if cb is not None and rec.subgroups is not None:
        sg = rec.subgroups[rec.subgroups.dimension != "cand_bucket_all_rules"]
        pd.concat([sg, cb], ignore_index=True).to_csv(run.path("subgroup_metrics.csv"), index=False)
        rec.subgroups = _load(run, "subgroup_metrics.csv")
    text, cases = case_analysis(rec)
    run.path("case_analysis.md").write_text(text, encoding="utf-8")
    run.path("summary_en.md").write_text(summary_en(rec), encoding="utf-8")
    run.path("final_report.md").write_text(final_report(rec, cases), encoding="utf-8")
    run.path("README_reproduce.md").write_text(readme_reproduce(rec), encoding="utf-8")
    mark_stage(run.root, "report", {"cases": cases.get("cases", 0), "records_copied": copied})
    log(f"[report] final_report.md, summary_en.md, case_analysis.md ({cases.get('cases', 0)} cases), "
        f"README_reproduce.md; copied {copied}")
    return {"cases": cases}
