#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py — render fig1-fig4 (PNG + SVG) from the packaged data.

Reads only files inside the package (no network, no model calls):
    results/tables/table1_main_results.csv   fig1, fig3 (judge score and in-KB rate with CIs)
    results/tables/table3_rag_lift.csv       fig2 (R-N lifts except the in-KB row)
    results/tables/table4_token_usage.csv    fig3 (mean net total tokens per answer)
    results/extra/intent_threshold_sweep.csv fig4
    results/scored.csv                       fig2 in-KB row (paired R-N per question)

The in-KB rate is the reply-level `in_kb_reply_ok` of table 1. Its paired CI uses
the table bootstrap: percentile, over questions, 10,000 resamples,
random.Random(42) per cell (harness/analyze.py:boot_ci).

The data-preparation functions (fig1_data ... fig4_data) use the standard library
only; rendering needs matplotlib.

Usage:
    python tools/make_figures.py [--results-dir results] [--out-dir results/figures]
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "harness"))

from analyze import MODELS, boot_ci  # noqa: E402

N_BOOT = 10000
SEED = 42
SHORT = {"Hy3": "Hy3", "Deepseek-V4.1-Flash": "DS-v4.1-F", "GLM-5.3": "GLM-5.3",
         "GLM-5.3-Flash": "GLM-5.3-F", "MiniMax-M3": "MiniMax-M3", "Kimi-K3": "Kimi-K3"}
COL_N, COL_R, COL_T, COL_DOWN = "#E69F00", "#0072B2", "#009E73", "#D55E00"
FIG_NAMES = ["fig1_grouped_bars", "fig2_rag_lift", "fig3_quality_vs_tokens",
             "fig4_threshold_sweep"]


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ data preparation

def fig1_data(results: Path) -> dict:
    """{'N'|'R': {model: {'judge': (mean, lo, hi), 'inkb': (mean, lo, hi)}},
    'T': {'judge': mean, 'inkb': mean}} from table 1."""
    out: dict = {"N": {}, "R": {}, "T": {}}
    for r in _rows(results / "tables" / "table1_main_results.csv"):
        cell = {k: (float(r[f"{c}_mean"]), float(r[f"{c}_ci_lo"]), float(r[f"{c}_ci_hi"]))
                for k, c in (("judge", "judge_avg"), ("inkb", "inkb_rate"))}
        if r["group"] == "T":
            out["T"] = {k: v[0] for k, v in cell.items()}
        else:
            out[r["group"]][r["model"]] = cell
    return out


def inkb_paired(results: Path) -> dict[str, tuple[float, float, float]]:
    """Per model: mean paired R-N difference of in_kb_reply_ok over the 80 questions
    and its bootstrap CI."""
    by: dict[tuple[str, str], dict[str, float]] = {}
    for r in _rows(results / "scored.csv"):
        if r["group"] in ("R", "N") and r["in_kb_reply_ok"] != "":
            by.setdefault((r["model_label"], r["qid"]), {})[r["group"]] = \
                1.0 if r["in_kb_reply_ok"] == "True" else 0.0
    out = {}
    for m in MODELS:
        diffs = [g["R"] - g["N"] for (ml, _q), g in by.items()
                 if ml == m and "R" in g and "N" in g]
        lo, hi = boot_ci(diffs, N_BOOT, random.Random(SEED))
        out[m] = (sum(diffs) / len(diffs), lo, hi)
    return out


def fig2_data(results: Path) -> list[tuple[str, list[tuple[str, float, float, float, bool]]]]:
    """[(row label, [(model, diff, ci_lo, ci_hi, significant) x 6])]; every row except
    the in-KB one is read from table 3."""
    t3 = {}
    for r in _rows(results / "tables" / "table3_rag_lift.csv"):
        lo, hi = (float(x) for x in re.findall(r"-?\d+\.\d+", r["95% CI"]))
        t3[(r["Model"], r["Metric"])] = (float(r["Mean Diff (R-N)"]), lo, hi,
                                         r["Significant"] == "yes")
    paired = inkb_paired(results)
    spec = [("Judge score (overall)", "judge_avg"),
            ("In-KB citation rate (reply-level)", None),
            ("Episode OK (scene)", "episode_ok"),
            ("All constraints (constraint)", "all_satisfy"),
            ("Frank rate (trap)", "rule_frank"),
            ("Fact OK (fact)", "fact_ok")]
    out = []
    for label, metric in spec:
        rows = []
        for m in MODELS:
            if metric is None:
                d, lo, hi = paired[m]
                rows.append((m, d, lo, hi, lo > 0 or hi < 0))
            else:
                rows.append((m, *t3[(m, metric)]))
        out.append((label, rows))
    return out


def fig3_data(results: Path) -> dict:
    """{'N'|'R': {model: (x net total tokens, y judge, lo, hi)}, 'T': judge mean}."""
    x = {(r["Model"], r["Group"]): float(r["Avg Net Total"])
         for r in _rows(results / "tables" / "table4_token_usage.csv")}
    f1 = fig1_data(results)
    out: dict = {"N": {}, "R": {}, "T": f1["T"]["judge"]}
    for g in ("N", "R"):
        for m in MODELS:
            y, lo, hi = f1[g][m]["judge"]
            out[g][m] = (x[(m, g)], y, lo, hi)
    return out


def fig4_data(results: Path) -> list[dict]:
    return [{"t": float(r["t"]), "easy": float(r["easy_rate"]),
             "hard": float(r["hard_rate"]), "misfire": float(r["misfire_rate"])}
            for r in _rows(results / "extra" / "intent_threshold_sweep.csv")]


# ------------------------------------------------------------------- rendering

def _style(plt):
    plt.rcParams.update({"font.size": 11, "axes.grid": False, "svg.fonttype": "path",
                         "svg.hashsalt": "page4-rag-eval", "savefig.dpi": 200})


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = []
    for ext, meta in (("png", {"Software": None}), ("svg", {"Date": None})):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, metadata=meta)
        paths.append(p)
    return paths


def _fig1(plt, results: Path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = fig1_data(results)
    fig, axes = plt.subplots(1, 2, figsize=(14.1, 5.9))
    panels = [("judge", "LLM-judge overall score (1-5)", (0, 4.7), d["T"]["judge"]),
              ("inkb", "In-KB citation rate (reply-level)", (0, 1.12), d["T"]["inkb"])]
    xs = list(range(len(MODELS)))
    for ax, (key, ylabel, ylim, t_line) in zip(axes, panels):
        for off, g, col in ((-0.2, "N", COL_N), (0.2, "R", COL_R)):
            m = [d[g][mod][key][0] for mod in MODELS]
            lo = [d[g][mod][key][1] for mod in MODELS]
            hi = [d[g][mod][key][2] for mod in MODELS]
            ax.bar([x + off for x in xs], m, width=0.38, color=col, edgecolor="white",
                   linewidth=0.6, zorder=2)
            ax.errorbar([x + off for x in xs], m,
                        yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]],
                        fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
        ax.axhline(t_line, color=COL_T, linestyle="--", linewidth=1.6, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=20, ha="right")
        ax.set_ylim(*ylim)
        if key == "judge":
            ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#b0b0b0", alpha=0.35, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    handles = [Patch(color=COL_N, label="N: LLM only"), Patch(color=COL_R, label="R: RAG"),
               Line2D([0], [0], color=COL_T, linestyle="--", linewidth=1.6,
                      label="T: system default")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.17, wspace=0.22)
    return fig


def _fig2(plt, results: Path):
    from matplotlib.patches import Patch
    groups = fig2_data(results)
    fig, ax = plt.subplots(figsize=(14.1, 9.4))
    ticks, labels, centers = [], [], []
    pitch = len(MODELS) + 1.3
    for gi, (_label, rows) in enumerate(groups):
        ys = []
        for j, (model, diff, lo, hi, sig) in enumerate(rows):
            y = gi * pitch + j
            ax.barh(y, diff, height=0.84, color=COL_R if diff >= 0 else COL_DOWN,
                    alpha=1.0 if sig else 0.45, edgecolor="white", linewidth=0.5, zorder=2)
            ax.errorbar(diff, y, xerr=[[diff - lo], [hi - diff]], fmt="none",
                        ecolor="black", elinewidth=0.9, capsize=2.2, zorder=3)
            ticks.append(y)
            labels.append(SHORT[model])
            ys.append(y)
        centers.append(sum(ys) / len(ys))
    ax.axvline(0, color="black", linewidth=1.0, zorder=4)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(len(groups) * pitch - 1.3 + 0.7, -0.7)
    for (label, _rows_), c in zip(groups, centers):
        ax.text(-0.115, c, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.97, 1.0)
    ax.set_xlabel("R − N paired difference (bootstrap 95% CI)")
    ax.grid(axis="x", color="#b0b0b0", alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=COL_R, label="increase (sig. solid / n.s. faded)"),
                       Patch(color=COL_DOWN, label="decrease (sig. solid / n.s. faded)")],
              loc="lower right", frameon=True)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.99, bottom=0.07)
    return fig


# label anchors for fig3: (text x, text y, horizontal alignment); text is joined to its
# marker with a thin leader line when it sits away from the marker
_F3_LABELS = {
    ("N", "Kimi-K3"): (398, 3.652, "right"),
    ("N", "GLM-5.3"): (590, 3.29, "right"),
    ("N", "GLM-5.3-Flash"): (668, 3.29, "left"),
    ("N", "MiniMax-M3"): (1465, 3.12, "center"),
    ("N", "Hy3"): (1650, 3.617, "left"),
    ("N", "Deepseek-V4.1-Flash"): (2200, 3.577, "left"),
    ("R", "Kimi-K3"): (655, 4.117, "right"),
    ("R", "GLM-5.3"): (790, 3.80, "left"),
    ("R", "GLM-5.3-Flash"): (790, 4.32, "left"),
    ("R", "MiniMax-M3"): (1035, 4.085, "right", True),
    ("R", "Deepseek-V4.1-Flash"): (1035, 3.935, "right", True),
    ("R", "Hy3"): (1290, 4.027, "left"),
}


def _fig3(plt, results: Path):
    from matplotlib.lines import Line2D
    d = fig3_data(results)
    fig, ax = plt.subplots(figsize=(10.6, 7.0))
    for m in MODELS:
        xn, yn, *_ = d["N"][m]
        xr, yr, *_ = d["R"][m]
        ax.annotate("", xy=(xr, yr), xytext=(xn, yn), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.0,
                                    shrinkA=7, shrinkB=7))
    for g, col, marker, face in (("N", COL_N, "o", "white"), ("R", COL_R, "s", COL_R)):
        for m in MODELS:
            x, y, lo, hi = d[g][m]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=marker, color=col,
                        markerfacecolor=face, markeredgewidth=1.6, markersize=8,
                        capsize=3, elinewidth=1.3, zorder=3)
            tx, ty, ha, *force = _F3_LABELS[(g, m)]
            name = SHORT[m] + ("*" if m == "MiniMax-M3" else "")
            far = bool(force) or abs(tx - x) / x > 0.12 or abs(ty - y) > 0.12
            ax.annotate(name, xy=(x, y), xytext=(tx, ty), ha=ha, va="center", fontsize=9.5,
                        color="#333333", zorder=4,
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7,
                                        shrinkA=2, shrinkB=7) if far else None)
    ax.axhline(d["T"], color=COL_T, linestyle="--", linewidth=1.6, zorder=1)
    ax.text(305, d["T"] + 0.035, "T: system default", color=COL_T, fontsize=10, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(300, 3000)
    ax.set_xticks([400, 600, 1000, 1500, 2000])
    ax.set_xticklabels(["400", "600", "1000", "1500", "2000"])
    ax.minorticks_off()
    ax.set_ylim(2.25, 4.45)
    ax.set_yticks([2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25])
    ax.set_xlabel("Mean net total tokens per answer (net input + output, log scale)")
    ax.set_ylabel("LLM-judge overall score (1-5)")
    ax.grid(color="#b0b0b0", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=COL_N, markerfacecolor="white",
               markeredgewidth=1.6, markersize=8, linestyle="none", label="N: LLM only"),
        Line2D([0], [0], marker="s", color=COL_R, markersize=8, linestyle="none",
               label="R: RAG"),
        Line2D([0], [0], color="#888888", marker=">", markersize=5, linewidth=1.0,
               label="N → R, same model")],
        loc="lower right", bbox_to_anchor=(0.99, 0.1), frameon=True)
    fig.text(0.012, 0.012, "* MiniMax-M3: net input comes from an unreliable regression "
                           "(R² = 0.28).", fontsize=9, color="#333333")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.14)
    return fig


def _fig4(plt, results: Path):
    rows = fig4_data(results)
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.8, 6.75))
    for key, col, marker, label in (
            ("easy", COL_R, "o", "Scene questions, easy (n=12)"),
            ("hard", COL_N, "s", "Scene questions, hard / paraphrased (n=12)"),
            ("misfire", COL_T, "^", "Non-scene questions misfired (n=56)")):
        ax.plot(t, [r[key] for r in rows], marker=marker, color=col, linewidth=2.4,
                markersize=7, label=label, zorder=3)
    ax.axvline(0.34, color="#888888", linestyle="--", linewidth=1.4, zorder=2)
    ax.text(0.3375, 0.62, "current t = 0.34", rotation=90, ha="right", va="center",
            fontsize=10.5, color="#555555")
    ax.set_xticks(t)
    # 0.34 and 0.35 are one step apart: the 0.34 label sits on a second line
    ax.set_xticklabels([("\n" if abs(x - 0.34) < 1e-9 else "") + f"{x:.2f}" for x in t])
    ax.set_xlim(0.085, 0.565)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Scene-intent threshold t (top scene score ≥ t)", labelpad=10)
    ax.set_ylabel("Trigger / misfire rate")
    ax.set_title("Scene-intent threshold sweep (real tokenizer & scorer, 80 queries)")
    ax.grid(color="#b0b0b0", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", bbox_to_anchor=(0.995, 0.5), fontsize=10, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.14)
    return fig


def render_all(results: Path, out_dir: Path) -> list[Path]:
    """Render fig1-fig4 as PNG + SVG into out_dir. Requires matplotlib."""
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover
        raise SystemExit("make_figures: matplotlib is required to render figures") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, fn in zip(FIG_NAMES, (_fig1, _fig2, _fig3, _fig4)):
        fig = fn(plt, results)
        paths += _save(fig, out_dir, name)
        plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render fig1-fig4 from packaged data (offline).")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/figures")
    args = ap.parse_args(argv)
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = PKG / results
    out_dir = Path(args.out_dir) if args.out_dir else results / "figures"
    if not out_dir.is_absolute():
        out_dir = PKG / out_dir
    for p in render_all(results, out_dir):
        print(f"figure -> {p}")
    print("FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
