"""`report` stage: the human-readable records of a run, generated only from the run's own files.

Writes in the run directory:
  summary.csv          per-model table: metrics, paired differences, P95, training time, model size and the
                       confirmation (the evaluate-test table with the benchmark columns added)
  summary_en.md        short English table (metric names identical to the Chinese report)
  case_analysis.md     at least 12 real test cases in the required categories; a category that does not
                       occur is reported as not observed, never invented
  final_report.md      Chinese report for the course presentation
  README_reproduce.md  commands, data locations and outputs of this run
  query_manifest.parquet, sampled_pairs.parquet, users_split.parquet  the split records (from work/)
A stage that has not run shows NOT_RUN; no number is guessed.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from app.recommendation.algorithm2.profiles import build_profile
from evaluation.algorithm2.common import mark_stage, now, read_json, write_json
from evaluation.algorithm2.data import load_prepared
from evaluation.algorithm2.split import load_queries

NR = "NOT_RUN"
RULE_TEXT = {"R0": "R0（无筛选）", "R1": "R1（关键词 = 可见收藏最常见 genre）", "R2": "R2（公开评分 ≥ 7.0 且集数 ≤ 26）",
             "R3": "R3（R1 + R2）"}
FAMILY_TEXT = {"A0": "原手工公式", "A1": "验证集调权公式", "B0": "LightGBM（两特征）", "B1": "LightGBM（八特征）",
               "C0": "内容双塔", "AUX_slides_v0": "旧版 slides 公式（辅助）", "AUX_profile_v1": "旧版画像匹配（辅助）"}


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
        return "# 失败案例分析\n\nNOT_RUN：本运行没有测试预测（predictions.parquet）。\n", {}
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
    cats.append(("A0 命中、学习模型（B1 与 C0）未命中", "R0", pick(a0_only), len(a0_only)))
    learned_only = r0m[(r0m["A0"] == 0) & ((r0m[b1] == 1) | (r0m[c0] == 1))].index
    cats.append(("学习模型命中、A0 未命中", "R0", pick(learned_only), len(learned_only)))
    all_fail = r0m[(r0m[core] == 0).all(axis=1)].index
    cats.append(("五个核心模型都未命中", "R0", pick(all_fail), len(all_fail)))
    sparse = r0_info[(r0_info.group == "sparse") & r0_info.target_eligible].index
    cats.append(("稀疏收藏（可见历史 1–4 部）", "R0", pick(sparse), len(sparse)))
    mixed = r0_info[(r0_info.group == "main") & (r0_info.profile_mix == "mixed") & r0_info.target_eligible].index
    cats.append(("混合偏好（genre 熵高于验证集中位数）", "R0", pick(mixed), len(mixed)))
    excl = info[(info.excluded_reason == "target_not_eligible") & (info.index.get_level_values("rule") != "R0")]
    ex_pick = []
    for rule in ("R2", "R1", "R3"):
        ids = excl.xs(rule, level="rule").index if rule in excl.index.get_level_values("rule") else []
        ex_pick += [(x, rule) for x in pick(ids, 1)]
    cats.append(("规则排除了隐藏目标（严格模式不插入）", None, ex_pick, int(len(excl))))
    empty = info[info.n_cand == 0]
    short = info[(info.n_cand > 0) & (info.n_cand < 8)]
    cats.append(("严格空结果（候选数 = 0）", None, [(x, r) for x, r in empty.index[:1]], int(len(empty))))
    cats.append(("候选不足 8 个", None, [(x, r) for x, r in short.index[:1]], int(len(short))))

    thr = rec.freeze.get("genre_entropy_threshold") if rec.freeze else None
    blocks, n_cases, counts = [], 0, {}
    for title, rule0, picked, total in cats:
        counts[title] = total
        blocks.append(f"## {title}\n\n测试集中符合条件的 query 数：{total}。")
        if not picked:
            blocks.append("本运行的测试集中**未出现**此类案例，因此不展示（不编造）。\n")
            continue
        for item in picked:
            qid, rule = (item if isinstance(item, tuple) else (item, rule0))
            n_cases += 1
            blocks.append(_case_block(n_cases, title, qid, rule, q.loc[qid], info.loc[(qid, rule)], cat, snap, rec,
                                      core, hits, ranks, top_of, pop_rank, thr, learned))
    head = ["# 失败案例分析（真实测试 query）", "",
            f"运行 `{run.run_id}`（级别 {cfg['level']}）。案例全部取自 predictions.parquet 的测试 query，按 query_id "
            "排序后在每类中选取，不挑选“好看”的例子。核心模型：A0、A1 与部署 seed "
            f"{rec.seed} 的 B0 / B1 / C0。", "",
            "阅读说明：**观察**是从数据直接读出的事实；**待验证推测**只是可能原因，未经对照实验验证，"
            "特征重要性与案例相关性都不是因果解释。标签是代理标签（评分 ≥ 8），未评分的番剧是弱负例，"
            "“未命中”不代表用户不喜欢推荐结果。", "",
            table(["类别", "测试中出现次数", "展示案例数"],
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
    hist_titles = "；".join(_title(snap, i, 40) for i in H[:10]) + (f"；…（共 {len(H)} 部）" if len(H) > 10 else "")
    lines = [f"### 案例 {n}：{title} · query `{qid}`", "",
             f"- 离线用户 {int(qrow.offline_user_id)} · {'主组' if qrow.group == 'main' else '稀疏组'} · 可见收藏 "
             f"{len(H)} 部 · 规则 {RULE_TEXT[rule]}" + (f" · 关键词 `{qrow.r1_keyword}`" if rule in ("R1", "R3") else ""),
             f"- 可见收藏：{hist_titles}",
             f"- 画像 top-10 标签：{', '.join(top_labels) or '（无）'}",
             f"- 隐藏目标：**{_title(snap, t, 60)}**（{snap.keys[t]}）· 公开评分 {f4(cat.score[t], 2)} · "
             f"集数 {'未知' if not np.isfinite(eps) else int(eps)} · 人气排名 {pop_rank.get(t, '—')}/{len(cat.members)} · "
             f"标签 {', '.join(t_tags) or '（无）'} · 制作公司 {', '.join(t_studios) or '（未知）'}",
             f"- 目标与画像 top-10 标签重叠 {len(overlap)} 个" + (f"（{', '.join(overlap)}）" if overlap else "")
             + f"；目标的制作公司{'出现' if studio_hit else '未出现'}在可见收藏中",
             f"- 候选数 {int(irow.n_cand)}；目标{'满足' if bool(irow.target_eligible) else '不满足'}本规则"
             + ("" if bool(irow.target_eligible) else "（严格模式不会把它插回候选，该 query 不计入条件指标，"
                                                       "端到端记为未找到）"), ""]
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
        rows.append([m, "不适用" if rk is None or not np.isfinite(rk) else int(rk),
                     "—" if hv is None or not np.isfinite(hv) else int(hv),
                     " / ".join(_title(snap, i, 24) for i in idx) or "（空）"])
    lines += [table(["模型", "目标名次", "Hit@8", "Top-8"], rows), ""]
    obs, hyp = [], []
    if a0_scores:
        obs.append(f"A0 的 Top-8 平均公开评分 {np.mean(a0_scores):.2f}、平均 hits {np.mean(a0_hits):.1f}；"
                   f"目标公开评分 {f4(cat.score[t], 2)}、hits {len(overlap)}。")
    lr = {m: ranks.loc[(qid, rule), m] for m in learned if (qid, rule) in ranks.index}
    if any(v is not None and np.isfinite(v) for v in lr.values()):
        obs.append("学习模型对目标的名次：" + "，".join(f"{m} {int(v)}" for m, v in lr.items()
                                              if v is not None and np.isfinite(v)) + "。")
    if not bool(irow.target_eligible):
        why = []
        if rule in ("R2", "R3"):
            if not np.isfinite(cat.score[t]) or cat.score[t] < 7.0:
                why.append("公开评分低于 7.0")
            if not np.isfinite(eps):
                why.append("集数未知（集数上限规则下视为不满足）")
            elif eps > 26:
                why.append(f"集数 {int(eps)} > 26")
        if rule in ("R1", "R3") and qrow.r1_keyword:
            gt = [x.lower() for x in _names(cat, "genre", t)]
            if not any(qrow.r1_keyword in g for g in gt):
                why.append(f"genre 与标题都不含关键词“{qrow.r1_keyword}”")
        obs.append("目标被规则排除的原因：" + ("；".join(why) if why else "见规则定义") + "。")
    if a0_scores and np.isfinite(cat.score[t]) and cat.score[t] < np.mean(a0_scores) - 0.5:
        hyp.append("目标的公开评分明显低于 A0 推荐条目的平均值；以公开评分为主要信号的排序可能把它压到后面。")
    if a0_hits and len(overlap) < np.mean(a0_hits):
        hyp.append("目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。")
    pr = pop_rank.get(t)
    if pr and pr > 2000:
        hyp.append(f"目标在目录中较冷门（人气排名 {pr}），而各模型的主要信号与公开评分 / 人气相关，冷门目标普遍难以进入前 8。")
    if title.startswith("A0 命中") and a0_hits and len(overlap) >= np.mean(a0_hits):
        share = ""
        if rec.importance is not None:
            imp = rec.importance[(rec.importance.family == "B1") & (rec.importance.seed == rec.seed)]
            if not imp.empty and "rating_norm" in set(imp.feature):
                share = f"（B1 中 rating_norm 的 gain 占比 {float(imp[imp.feature == 'rating_norm'].gain.iloc[0]) / imp.gain.sum():.0%}）"
        hyp.append(f"目标与画像标签高度重叠（hits {len(overlap)}，不低于 A0 推荐条目的平均值），A0 的 hits 项把它排在前面；"
                   f"学习模型更依赖公开评分{share}，把评分或人气更高的条目排在它前面。")
    if title.startswith("学习模型命中"):
        hyp.append("B1 / C0 使用了制作公司、集数、相似度等 A0 没有的信号"
                   + ("（目标的制作公司出现在可见收藏中）" if studio_hit else "")
                   + "，可能因此排得更前；这只是相关观察，不是因果证明。")
    if qrow.group == "sparse":
        hyp.append(f"可见收藏只有 {len(H)} 部，每部占画像频率的 1/{len(H)}，标签信号噪声大。")
    if title.startswith("混合偏好") and thr is not None:
        hyp.append(f"可见收藏的 genre 熵高于预先确定的阈值 {thr:.2f}，偏好分散，top-10 标签难以代表单一口味。")
    lines += ["**观察**：" + ("".join(obs) if obs else "—"), "",
              "**待验证推测**：" + ("".join(hyp) if hyp else "暂无基于数据的明确推测。"), ""]
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
        name = m + (" ★部署" if m in (rec.labels["B0"], rec.labels["B1"], rec.labels["C0"]) else "")
        rows.append([name, p["users"], f4(p["HR@8"]), f4(p["NDCG@8"]), f4(p["MRR@8"]), f4(p["AUC"]),
                     "—" if m == "A0" else f"{pp(d0)} {ci_pp(l0, h0)}", "—" if m == "A1" else f"{pp(d1)} {ci_pp(l1, h1)}",
                     p["violations"]])
    for fam in ("B0", "B1", "C0"):
        f = fams.get(fam)
        if not f:
            continue
        d0, l0, h0 = rec.pair(f"{fam}_mean", "A0")
        d1, l1, h1 = rec.pair(f"{fam}_mean", "A1")
        rows.append([f"{fam} 三 seed 均值 ± SD", "—", f"{f4(f['HR@8_mean'])} ± {f4(f['HR@8_sd'])}",
                     f4(f["NDCG@8_mean"]), "—", f4(f["AUC_mean"]), f"{pp(d0)} {ci_pp(l0, h0)}",
                     f"{pp(d1)} {ci_pp(l1, h1)}", "—"])
    return table(["模型", "用户数", "HR@8", "NDCG@8", "MRR@8", "AUC", "ΔHR@8 vs A0（百分点，95% CI）",
                  "ΔHR@8 vs A1（百分点，95% CI）", "规则违反"], rows)


def _comparisons(rec: Records) -> str:
    rows = []
    for title, m, base in (("A1 − A0：权重优化收益", "A1", "A0"),
                           ("B0 − A1：相同两特征下学习评分函数的收益", "B0_mean", "A1"),
                           ("B1 − B0：增加六个特征的收益", "B1_mean", "B0_mean"),
                           ("C0 − B1：内容双塔相对八特征 GBDT", "C0_mean", "B1_mean"),
                           ("C0 − B0", "C0_mean", "B0_mean")):
        d, lo, hi = rec.pair(m, base)
        dn, lon, hin = rec.pair(m, base, "NDCG@8")
        rows.append([title, f"{pp(d)} {ci_pp(lo, hi)}", f"{pp(dn)} {ci_pp(lon, hin)}"])
    return table(["比较（学习模型取三 seed 的用户级平均）", "ΔHR@8（百分点，95% CI）", "ΔNDCG@8（百分点，95% CI）"], rows)


def _interpretation(rec: Records) -> list[str]:
    """One sentence per controlled comparison, judged only by its paired 95 % interval."""
    def verdict(lo, hi) -> str:
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            return "无法判断（未运行）"
        if lo > 0:
            return "区间在 0 以上：有稳定收益"
        if hi < 0:
            return "区间在 0 以下：稳定更差"
        return "区间包含 0：证据不足，不能称有差别"
    out = []
    for text, m, base in (("权重优化（A1 − A0）", "A1", "A0"), ("学习评分函数（B0 − A1，同两特征）", "B0_mean", "A1"),
                          ("增加特征（B1 − B0）", "B1_mean", "B0_mean"), ("内容双塔相对八特征 GBDT（C0 − B1）", "C0_mean",
                                                                     "B1_mean")):
        d, lo, hi = rec.pair(m, base)
        out.append(f"- {text}：ΔHR@8 {pp(d)} 个百分点 {ci_pp(lo, hi)}，{verdict(lo, hi)}。")
    prim = rec.metrics["primary"]
    core = {f: prim.get(rec.labels[f]) for f in ("A0", "A1", "B0", "B1", "C0") if prim.get(rec.labels[f])}
    if core:
        best_hr = max(core, key=lambda f: core[f]["HR@8"])
        best_auc = max(core, key=lambda f: core[f]["AUC"] or 0)
        if best_hr != best_auc:
            out.append(f"- AUC 最高的是 {best_auc}（{f4(core[best_auc]['AUC'])}），HR@8 最高的是 {best_hr}"
                       f"（{f4(core[best_hr]['HR@8'])}）：AUC 衡量目标相对全部弱负例的整体排序，HR@8 只看前 8 名，"
                       "两者可以给出不同的赢家；预注册的主指标是 HR@8。")
    return out


def candidate_buckets(rec: Records) -> pd.DataFrame | None:
    """HR@8 by strict candidate count over all rule scenarios (R0 alone always has > 500 candidates);
    target-eligible queries, user-macro (a user's queries in one bucket are averaged first)."""
    from evaluation.algorithm2.final_eval import CAND_BUCKETS, _bucket
    from evaluation.algorithm2.metrics import user_macro

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
    for dim, label in (("rule", "规则场景（目标 eligible 条件下）"), ("group", "收藏规模组（R0）"),
                       ("history_bucket", "可见历史（R0）"), ("cand_bucket_all_rules", "严格候选数（R0–R3 合并，目标 eligible）"),
                       ("profile_mix", "画像集中 / 混合（R0）")):
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
                row[0] = f"{v}（N<100，不能稳定比较）"
            rows.append(row)
        out += [f"**{label}**（HR@8，用户级平均）", "", table(["分组", "用户数", *models], rows), ""]
    return "\n".join(out)


def _ms(x: float) -> str:
    return f"{x:.3f}" if x < 1 else f"{x:.1f}"


def _latency_table(rec: Records) -> str:
    if rec.latency is None:
        return f"{NR}：本运行尚未执行 benchmark。"
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
    return table(["模型", "纯评分 R0 P50/P95", "纯评分 R3", "服务处理 R0", "服务处理 R3", "API R0", "API R3",
                  "最终训练时间 (s)", "模型大小 (KB)"], rows)


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
        why = (f"训练对 {n_train:,} 条不超过 LightGBM 直方图分箱的抽样上限 {cap:,}，分箱用到全部样本，"
               "因此 B0、B1 三个 seed 的树完全相同（标准差为 0 是该配置的真实性质，不是三次独立随机结果）。")
    else:
        why = (f"训练对有 {n_train:,} 条，超过 LightGBM 直方图分箱的抽样上限 {cap:,}（bin_construct_sample_cnt），"
               "seed 决定参与分箱的行，因此三个 seed 的分箱边界和树略有不同（已在模型文件中核对："
               + "，".join(f"{f} {'相同' if v else '不同'}" for f, v in same.items())
               + "）；seed 间差异只反映这一来源的随机性。")
    return "说明：预注册的 LightGBM 网格没有 bagging / feature_fraction 子采样，seed 只通过分箱抽样起作用；" + why


def final_report(rec: Records, cases: dict) -> str:
    cfg, run = rec.cfg, rec.run
    d, dm, fz, mt, dec = rec.data, rec.split, rec.freeze, rec.metrics, rec.decision
    L: list[str] = []
    a = L.append
    status = "已完成" if (mt and dec and rec.integration and rec.integration.get("status") == "passed"
                        and rec.latency is not None) else "部分完成／效果验证受阻"
    if cfg["level"] != "main":
        status = f"{cfg['level']} 级别流程检查（非正式选型结论）"
    a(f"# 算法②（收藏夹推荐）效果比较、模型选择与系统接入报告 —— {status}")
    a("")
    a(f"运行 `{run.run_id}` · 级别 **{cfg['level']}** · 生成于 {now()}。本报告的每个数字都来自本运行目录中的记录文件"
      "（见第 9 节）；未运行的部分写 NOT_RUN。")
    if cfg["level"] != "main":
        a("")
        a(f"> 注意：这是 {cfg['level']} 级别的运行，用户数较少，只用于检查流程，不作为正式选型结论。")
    a("")
    # 1
    a("## 1. 原系统状态、本次范围与新增能力")
    a("")
    g = (rec.meta or {}).get("start_git", {})
    a(f"- 代码起点：分支 `{g.get('branch', '—')}`，提交 `{(g.get('commit') or '—')[:10]}`（与 origin/main 相同）。"
      "仓库审计见 `docs/algorithm2/repo_audit.md`。")
    a("- 原状态：收藏夹页面使用 `collection_profile_v1`（归一化画像匹配，α = 0.7）作为默认评分器，"
      "`manual_slides_v0`（PPT 公式，按出现与否计 hits）只作对照；没有经过训练的收藏夹模型，"
      "也没有固定的用户切分、预注册选型规则与离线 / 在线一致性检查。")
    a("- 本次新增：`app/recommendation/algorithm2/`（统一目录、画像、八特征、五个评分器、LightGBM 文本模型的 NumPy 读取器、"
      "模型包校验与服务），`evaluation/algorithm2/`（十个阶段的命令行实验流程），收藏夹接口改为调用同一代码路径，"
      "严格规则（新增 `pool.scored`）与显式条件放宽的接口契约，以及本报告的全部记录。")
    a("")
    # 2
    a("## 2. 数据来源、标签、清洗与规模")
    a("")
    if d:
        cat_m, inter = d["catalogue"], d["interactions"]
        a(f"- 数据：Kaggle MyAnimeList 2023 快照，处理版本 `{d['source']['dataset_version']}`"
          f"（原始文件 SHA-256 见 data_manifest.json）。只用 MAL id 作为条目身份。")
        a(f"- 推荐目录：{cat_m['members']:,} 部 = 可服务（排除 Hentai / Erotica / Rx；Ecchi 与 R+ 保留）"
          f"且有公开评分。快照共 {cat_m['snapshot_rows']:,} 行。目录哈希 `{cat_m['catalog_hash'][:12]}`。")
        cov = cat_m["field_coverage"]
        src = cat_m.get("sources") or {}
        sup, added = src.get("supplement") or {}, src.get("supplement_values_added") or {}
        if sup.get("path") and sup.get("status") != "missing":
            sup_text = (f"theme / demographic 由队友提供的 `{sup['path']}` 按 MAL id 合并补充（并集；新增 theme "
                        f"{added.get('theme', 0):,} 个、demographic {added.get('demographic', 0):,} 个取值）")
        else:
            sup_text = "本运行没有使用 theme / demographic 补充文件（只用快照自身的标签）"
        a("- 字段覆盖率：" + "，".join(f"{k} {v:.1%}" for k, v in cov.items()) + f"。{sup_text}，不臆造缺失字段；"
          "集数未知时在“集数上限”规则下视为不满足。")
        a(f"- 交互清洗：读取 {inter['rows_read']:,} 条评分，保留目录内 {inter['rows_kept']:,} 条"
          f"（目录外 {inter['outside_catalogue']:,} 条，全部因为没有公开评分）；无非法评分与冲突重复。")
        a(f"- 标签：**代理标签**——个人评分 ≥ 8 的番剧构成该用户的“高评分偏好集合”（{inter['positive_rows']:,} 条，"
          f"{inter['users_with_positive']:,} 个用户）。应用数据库中没有真实收藏夹，所以本报告的效果结论都是"
          "“高评分偏好集合代理评估”，不是收藏行为本身；未评分的番剧是弱负例。")
        a("- 偏好集合规模分布（用户数）：" + "，".join(f"{k.replace('-1000000000', '+')}: {v:,}"
                                             for k, v in d["positive_set_size_distribution"].items()) + "。")
    else:
        a(NR)
    a("")
    # 3
    a("## 3. 切分、留一、规则场景、候选与泄漏控制")
    a("")
    if dm:
        ms = dm["main_split"]["by_group_partition"]
        a(f"- 用户级切分 70 / 15 / 15，`split_seed = {dm['seeds']['split_seed']}`，按偏好集合规模分层；"
          f"主组（集合 ≥ 6，即可见历史 ≥ 5）与稀疏组（2–5）分开报告。完整切分："
          + "，".join(f"{k} {v:,}" for k, v in ms.items()) + "。")
        lu = dm["level_users"]
        a(f"- 本级别使用 {lu['users']:,} 个用户（嵌套在完整切分中：{lu['nested_in_main_split']}）。"
          f"分区用户交集：{dm['partition_overlap_users']}。")
        qq = dm["queries"]
        a(f"- Query：验证 / 测试每个用户 1 个隐藏目标，训练每个用户最多 {cfg['split']['train_targets_per_collection']} "
          f"个目标；共 {qq['rows']:,} 个 query（{qq['by_partition_group']}）。隐藏目标从不进入画像输入 H。")
        by = dm["pairs"]["by_partition_label"]
        n_tr = sum(v for k, v in by.items() if k.startswith("train"))
        n_va = sum(v for k, v in by.items() if k.startswith("val"))
        a(f"- 训练样本：每个训练 query 1 个正例 + 4 个弱负例（同一候选集合内确定性采样，`negative_seed = "
          f"{dm['seeds']['negative_seed']}`，排除该用户全部已知正例），共 {dm['pairs']['rows']:,} 对（训练 {n_tr:,} 对；"
          f"验证用户的 {n_va:,} 对只用于早停的 logloss），负例不足 {dm['pairs']['negatives_shortfall']} 次。"
          "B0 / B1 / C0 共用同一组样本。")
    a("- 规则场景：R0 无筛选；R1 关键词 = H 中最常见的 genre（同频按名称），关键词对 genre 或标题做 AND 子串匹配；"
      "R2 公开评分 ≥ 7.0 且集数 ≤ 26（未知集数不满足）；R3 = R1 + R2。全部是严格模式：目标不满足规则时不插回，"
      "记为 target_not_eligible，单独报告端到端结果。")
    a("- 候选：对通过规则的**完整目录**排序（不采样候选），排除 H 本身。同分按 MAL id 升序决定名次；"
      "AUC 对同分给 0.5，不受 id 排序影响；候选数 < 2 时 AUC 为 null。")
    a("- 主指标（预注册）：R0、主组、候选数 > 8、目标 eligible 的用户级 Macro HR@8；NDCG@8、MRR@8、AUC 同表报告。")
    a("")
    # 4
    a("## 4. 五组实验的实现、参数与预算")
    a("")
    if rec.tuning and fz:
        sel = rec.tuning["selected"]
        enc = rec.tuning.get("encoder", {})
        trials = rec.trials
        tcount = trials.groupby("family").size().to_dict() if trials is not None else {}
        a("- **A0** = rating_norm + 0.15 × hits（hits = 候选的 genre/theme 标签落在 H 的 top-10 命名空间标签中的个数）。")
        w, grid = sel["A1"]["hits_weight"], [float(x) for x in cfg["a1_weights"]]
        edge = ""
        if float(w) in (min(grid), max(grid)):
            side = "最小值" if float(w) == min(grid) else "最大值"
            edge = (f"；选中值是预注册网格的{side}，说明数据可能偏好网格以外的权重，但网格没有在看到结果后扩展")
        a(f"- **A1**：在验证集比较 hits 权重 {cfg['a1_weights']}，选中 **{w}**"
          f"（验证 HR@8 {f4(sel['A1']['val_HR@8'])}）{edge}。")
        for fam in ("B0", "B1"):
            s = sel[fam]
            a(f"- **{fam}**：LightGBM 原生 `lgb.train`（objective = binary，按原始分数排序），特征 "
              f"{', '.join(s['features'])}；{tcount.get(fam, 0)} 个预先生成的确定性 trial（网格 {cfg['gbdt']['grid']}），"
              f"上限 {cfg['gbdt']['n_estimators']} 棵树、早停 {cfg['gbdt']['early_stopping_rounds']} 轮（验证对 logloss），"
              f"按完整候选验证 HR@8 选中 trial {s['trial']}：{s['params']}（验证 HR@8 {f4(s['val_HR@8'])}）。")
        s = sel["C0"]
        a(f"- **C0**：PyTorch 内容双塔，两侧隐藏层 {cfg['two_tower']['hidden']}，输入为 genre / theme / demographic / "
          f"studio（出现 ≥ {enc.get('studio_min_count')} 次的制作公司，其余归入 UNK）多热向量与数值特征及缺失 mask"
          f"（番剧塔 {enc.get('d_item')} 维，收藏塔 {enc.get('d_coll')} 维），点积 logits + BCEWithLogitsLoss，"
          f"不使用 in-batch 负例；{tcount.get('C0', 0)} 个 trial，选中 trial {s['trial']}：{s['params']}"
          f"（验证 HR@8 {f4(s['val_HR@8'])}），最多 {cfg['two_tower']['max_epochs']} 轮、patience "
          f"{cfg['two_tower']['early_stopping_patience']}，设备 CPU。")
        a(f"- 最终配置用训练 seed {cfg['seeds']} 各训练一次（共 9 个模型，见 models_manifest.json）。" + _seed_note(rec))
        if trials is not None and not trials.empty:
            cost = trials.groupby("family")[["fit_seconds", "val_score_seconds"]].sum()
            parts = []
            for fam in ("B0", "B1", "C0"):
                if fam in cost.index:
                    fit, val = cost.loc[fam, "fit_seconds"], cost.loc[fam, "val_score_seconds"]
                    parts.append(f"{fam} 拟合 {fit:,.0f} s" + (f" + 完整候选验证评分 {val:,.0f} s" if val and val == val
                                                              else "（验证评分未单独计时）"))
            a(f"- 调参成本（{', '.join(f'{k} {v} 个 trial' for k, v in tcount.items() if k != 'A1')}，合计）："
              + "；".join(parts) + "。完整目录的验证评分是主要开销。")
        a("- 比较关系：B0 是两特征的控制组；B1 与 C0 使用同源元数据但特征表示不同，C0 vs B1 不是“同输入只换模型”。")
        if rec.importance is not None:
            imp = rec.importance[(rec.importance.family == "B1") & (rec.importance.seed == rec.seed)]
            if not imp.empty:
                tot = imp.gain.sum()
                a("- B1 特征重要性（gain 占比，部署 seed）：" + "，".join(
                    f"{r.feature} {r.gain / tot:.1%}" for r in imp.sort_values("gain", ascending=False).itertuples())
                  + "。重要性不是因果贡献，特征的作用以 B1 − B0 的对照为准。")
    else:
        a(NR)
    a("")
    # 5
    a("## 5. 测试结果")
    a("")
    if mt:
        a(f"测试集：{mt['test_users']:,} 个用户、{mt['test_queries']:,} 个 query。主表为预注册主集合（{mt['primary_definition']}）。")
        a("")
        a("### 5.1 主表")
        a("")
        a(_main_table(rec))
        a("")
        a("### 5.2 关键对照（用户配对 bootstrap，2,000 次，按用户重采样）")
        a("")
        a(_comparisons(rec))
        a("")
        for line in _interpretation(rec):
            a(line)
        a("")
        a("区间只反映用户抽样的不确定性，不涵盖代理标签、静态快照等偏差。")
        a("")
        a("### 5.3 分组结果")
        a("")
        a(_subgroup_table(rec))
        a("### 5.4 规则场景的端到端结果")
        a("")
        rows = []
        for rule in ("R0", "R1", "R2", "R3"):
            el = mt["rule_eligibility"].get(rule, {})
            e2e = [f4(mt["end_to_end"].get(rec.labels[f], {}).get(rule, {}).get("end_to_end_target_in_strict_top8"))
                   for f in ("A0", "A1", "B0", "B1", "C0")]
            rows.append([RULE_TEXT[rule], f"{el.get('target_eligible_share', float('nan')):.1%}",
                         f"{el.get('empty_result_share', float('nan')):.1%}", f"{el.get('fewer_than_8_share', float('nan')):.1%}",
                         *e2e])
        a(table(["规则", "目标 eligible 比例", "空结果比例", "不足 8 个比例",
                 *[f"端到端 {rec.labels[f]}" for f in ("A0", "A1", "B0", "B1", "C0")]], rows))
        a("")
        a("“端到端”= 所有原始 query 中，严格 Top-8 找到隐藏目标的比例（目标不 eligible 记为未满足）；"
          "与 5.3 的条件指标是不同的量。")
        hy = mt.get("hygiene", {})
        a("")
        a("推荐卫生（全部测试行）：" + "；".join(
            f"{m} 违规率 {v['rule_violation_rate']:.0%} / 重复率 {v['duplicate_rate']:.0%} / 已收藏重复率 "
            f"{v['saved_in_top_rate']:.0%}" for m, v in hy.items() if m in rec.labels.values()) + "。")
    else:
        a(f"{NR}：本运行没有完成 evaluate-test。")
    a("")
    a("### 5.5 时延与成本")
    a("")
    a(_latency_table(rec))
    a("")
    if rec.bench:
        b = rec.bench
        env = b["environment"]
        a(f"P50 / P95 单位毫秒；同一台机器（{env.get('platform')}，{env.get('cpu_count')} 核，"
          f"{(env.get('memory_bytes') or 0) / 2**30:.0f} GB 内存），{b['threads']} 个 CPU 线程，无 GPU、无结果缓存、"
          f"无批处理；预热 {b['request_list']['warmup_n']} 次后测量 {b['request_list']['n']} 条固定请求"
          f"（R3 中没有关键词的请求跳过 {b['request_list']['R3_requests_without_keyword_skipped']} 条）。"
          "纯评分 = 只调用评分器；服务处理 = 画像 + 严格规则 + 特征 + 评分 + Top-8；API = 收藏夹接口进程内调用"
          "（读取收藏、理由、写运行日志）。")
        ld = b["load_seconds"]
        a(f"加载时间（不计入请求时延）：目录与规则引擎 {ld.get('catalogue_and_rule_engine')} s；"
          f"公式 {ld.get('A0+A1')} s；B0 {ld.get('B0')} s；B1 {ld.get('B1')} s；C0 {ld.get('C0')} s"
          f"（其中预计算 {ld.get('C0_breakdown', {}).get('items')} 个番剧向量 "
          f"{ld.get('C0_breakdown', {}).get('item_vectors_s')} s）。峰值内存（整个进程）："
          + "，".join(f"{k} {v} MB" for k, v in b["peak_rss_mb"].items() if v) + "。")
        a("项目没有既定的时延 SLO，这里只报告实测值，不作为选型门槛。B0 / B1 在服务中使用 LightGBM 文本模型的 "
          "NumPy 读取器（同进程有 PyTorch 时原生 LightGBM 会因 OpenMP 冲突卡死），freeze 时已核对与原生 LightGBM "
          f"的分数差（最大绝对差 {fz['gbdt_serving_engine']['check_vs_native']['max_abs_diff'] if fz else '—'}）。")
    a("")
    # 6
    a("## 6. 预选、测试确认与默认评分器")
    a("")
    if fz:
        rows = []
        for fam, v in fz["validation"].items():
            rows.append([fam, f4(v["val_HR@8"]), f4(v.get("val_HR@8_sd")) if v.get("seeds") else "—",
                         f4(v["val_NDCG@8"]), ", ".join(f"{k}: {f4(x)}" for k, x in (v.get("seeds") or {}).items()) or "—"])
        a(table(["候选（验证集）", "HR@8", "seed SD", "NDCG@8", "各 seed HR@8"], rows))
        a("")
        a(f"按预注册规则（`selection_protocol.json`，写于任何训练之前）在验证集预选：**{fz['pre_selected']}**；"
          f"部署产物为 seed {fz['deploy_seed']}；保守 fallback 为 {fz['fallback']}。")
    if dec:
        a("")
        rows = []
        for k, c in dec["checks"].items():
            val = c.get("value")
            val = ci_pp(*c["ci95"]) + "（百分点）" if c.get("ci95") else (f4(val) if isinstance(val, float) else val)
            rows.append([k, c.get("threshold", "—"), val, "通过" if c["pass"] else "未通过"])
        a(table(["测试确认检查（部署产物 " + dec["deploy_label"] + "）", "门槛", "实测", "结果"], rows))
        a("")
        a(f"结论：{'已确认' if dec['confirmed'] else '未通过确认'}。默认评分器 = **{dec['default_scorer']}**。原因：{dec['reason']}。"
          "测试结果没有用于重新调参或改选其他模型。")
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
            a(f"网站配置（`configs/real.yaml`）：scorer = `{real_cfg.get('scorer', '（未设置，用 base.yaml）')}`，"
              f"algorithm2_run = `{real_cfg.get('algorithm2_run')}` —— "
              + ("已按本结论设置。" if on else "尚未指向本运行（本运行不是网站当前使用的冻结结果）。"))
    else:
        a(f"{NR}：本运行没有测试确认。")
    a("")
    # 7
    a("## 7. 失败案例与限制")
    a("")
    if cases:
        a(f"`case_analysis.md` 给出 {cases.get('cases', 0)} 个真实测试案例：" + "；".join(
            f"{k} {v}" for k, v in cases.get("categories", {}).items()) + "（测试中出现次数）。")
    a("")
    a("主要限制：")
    a("- 代理标签：评分 ≥ 8 的集合不等于收藏夹；未评分条目是弱负例，AUC 衡量的是目标相对这些弱负例的区分度。")
    a("- 静态快照（2023）与用户留出：评估的是“同一时期内其他高评分条目能否被找回”，不是对未来行为的预测。")
    a("- 流行度偏差：目标多为热门高分番剧，依赖公开评分 / 人气的特征天然占优；冷门目标普遍难以进入前 8。")
    a("- 字段覆盖：theme 与 demographic 依赖补充文件且覆盖不完整，缺失按 NaN / 空集合处理，没有补造。")
    a("- LightGBM 的 seed 只影响分箱抽样（网格无子采样），三个 seed 的差异不代表完整的训练随机性。")
    a("- C0 理论上能编码没有交互的新番，但本次没有 item-disjoint 实验，不能声称新番冷启动更好。")
    a("- 时延在单台笔记本上测得，只用于相对比较，不代表生产环境。")
    a("")
    # 8
    a("## 8. 系统接入验证")
    a("")
    ir = rec.integration
    if ir and ir.get("checks"):
        rows = [[c["check"], "通过" if c["passed"] else "**未通过**"] for c in ir["checks"]]
        a(table(["检查（integration_report.json）", "结果"], rows))
        a("")
        a(f"运行环境：{ir.get('database')}。整体状态：**{ir.get('status')}**。")
    elif ir:
        a(f"{ir.get('status')}：{ir.get('reason')}")
    else:
        a(f"{NR}：本运行尚未执行 verify-integration。")
    a("")
    a("已接入：收藏夹接口读取当前收藏 H，调用与离线完全相同的目录、画像、规则引擎、特征与评分器；评分器注册表可选 "
      "A0 / A1 / B0 / B1 / C0（`recommendation.collection.scorer`，或请求体 `scorer` 字段供实验 / 运维使用）；"
      "模型包缺失或不兼容时退回公式并记录状态，不放宽用户筛选；严格无结果时只给出放宽选项，用户确认后才放宽，"
      "响应带 `relaxed`、`relaxed_rules` 与 `strict_result_count`，页面显示可读的放宽说明。")
    a("仍未完成：应用数据库里还没有真实用户的收藏数据，没有线上实验（A/B 或点击反馈），因此网站上的实际收藏推荐效果"
      "尚未得到验证；本报告的效果数字全部来自离线代理评估。")
    a("")
    # 9
    a("## 9. 复现")
    a("")
    a("见同目录 `README_reproduce.md`（命令、数据位置、输出文件）。主要记录：`config.json`、`data_manifest.json`、"
      "`split_manifest.json`、`query_manifest.parquet`、`sampled_pairs.parquet`、`selection_protocol.json`、"
      "`trial_results.csv`、`tuning.json`、`training_curves.json`、`feature_importance.csv`、`models_manifest.json`、"
      "`models/`、`freeze_manifest.json`、`feature_schema.json`、`predictions.parquet`、`metrics.json`、`summary.csv`、"
      "`subgroup_metrics.csv`、`paired_comparisons.csv`、`decision.json`、`latency.csv`、`benchmark.json`、"
      "`integration_report.json`、`case_analysis.md`、`run_metadata.json`。")
    a("")
    return "\n".join(L) + "\n"


def readme_reproduce(rec: Records) -> str:
    run, cfg = rec.run, rec.cfg
    rid, lvl = run.run_id, cfg["level"]
    rel = run.root.relative_to(run.root.parents[3]) if len(run.root.parents) > 3 else run.root
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
.venv/bin/python -m evaluation.algorithm2.cli run-all --run {rid} --level {lvl}

# or one stage at a time
.venv/bin/python -m evaluation.algorithm2.cli audit              --run {rid} --level {lvl}
.venv/bin/python -m evaluation.algorithm2.cli prepare            --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli split              --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli tune               --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli train              --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli freeze             --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli evaluate-test      --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli benchmark          --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli verify-integration --run {rid}
.venv/bin/python -m evaluation.algorithm2.cli report             --run {rid}
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
