"""Build a clearly labelled, post-hoc high-consistency showcase subset.

This artifact is for presentation examples only. It is selected from already
scored MAL test targets using their saved truths and predictions, so it must
never replace the frozen unbiased test summary or drive model selection.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, read_json, read_rows, sha256, utc_now, write_json, write_rows

METHODS = ("A", "B", "C", "D")
SHOWCASE_SIZE = 40


def _summary(rows):
    result = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        result[method] = {
            "n_targets": len(method_rows),
            "coverage_at_5": float(np.mean([row["coverage_at_5"] for row in method_rows])),
            "ndcg_at_5": float(np.mean([row["ndcg_at_5"] for row in method_rows])),
            "hit_rate_at_5": float(np.mean([row["hit_rate_at_5"] for row in method_rows])),
            "precision_at_5": float(np.mean([row["precision_at_5"] for row in method_rows])),
        }
    return result


def build(size=SHOWCASE_SIZE):
    optimized_metrics_path = ROOT / "results/per_query_metrics.jsonl"
    baseline_metrics_path = ROOT / "results/baseline/per_query_metrics.jsonl"
    optimized_rows = [row for row in read_rows(optimized_metrics_path)
                      if row["source"] == "mal_recommendations" and row["split"] == "test"]
    baseline_rows = [row for row in read_rows(baseline_metrics_path)
                     if row["source"] == "mal_recommendations" and row["split"] == "test"]
    grouped = defaultdict(dict)
    for row in optimized_rows:
        grouped[row["target_id"]][row["method"]] = row
    if len(grouped) < size or any(set(rows) != set(METHODS) for rows in grouped.values()):
        raise ValueError("The frozen MAL test results do not contain every method for enough targets")
    # Mean utility across all four methods avoids selecting for one method only.
    ranked = []
    for target_id, rows in grouped.items():
        utility = float(np.mean([rows[method]["coverage_at_5"] + rows[method]["ndcg_at_5"]
                                 for method in METHODS]))
        ranked.append((utility, int(target_id)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = ranked[:size]
    chosen_ids = {target_id for _, target_id in chosen}
    labels = [row for row in read_rows(ROOT / "data/mal_ground_truth.jsonl")
              if row["source"] == "mal_recommendations" and row["split"] == "test"
              and row["target_id"] in chosen_ids]
    if len(labels) != size:
        raise ValueError("Showcase labels do not match the selected target count")
    labels.sort(key=lambda row: row["target_id"])
    target_rows = []
    rank_by_id = {target_id: rank for rank, (_, target_id) in enumerate(chosen, 1)}
    score_by_id = {target_id: utility for utility, target_id in chosen}
    for label in labels:
        target_rows.append(dict(target_id=label["target_id"], split="showcase", source_split="test",
                                source_sampling_group=label["sampling_group"], selection_rank=rank_by_id[label["target_id"]],
                                selection_utility=score_by_id[label["target_id"]],
                                selection_rule="top 40 by mean over A/B/C/D of coverage@5 + NDCG@5 on saved MAL test predictions",
                                posthoc_truth_selection=True))
    output = ROOT / "results/showcase"
    output.mkdir(parents=True, exist_ok=True)
    write_rows(ROOT / "data/showcase_targets.jsonl", target_rows)
    write_rows(output / "ground_truth_used.jsonl", labels)

    all_rows = {}
    for variant, path in (("baseline", baseline_metrics_path), ("optimized", optimized_metrics_path)):
        current = [row for row in read_rows(path) if row["source"] == "mal_recommendations"
                   and row["split"] == "test" and row["target_id"] in chosen_ids]
        if len(current) != size * len(METHODS):
            raise ValueError(f"Unexpected {variant} showcase metric count")
        for row in current:
            all_rows[(variant, row["target_id"], row["method"])] = row
    write_rows(output / "per_query_metrics.jsonl", [all_rows[key] for key in sorted(all_rows)])

    predictions = []
    for variant, path in (("baseline", ROOT / "results/baseline/predictions.jsonl"),
                          ("optimized", ROOT / "results/predictions.jsonl")):
        for row in read_rows(path):
            if row["target_id"] in chosen_ids:
                predictions.append(dict(variant=variant, **row))
    predictions.sort(key=lambda row: (row["variant"], row["target_id"], row["method"]))
    write_rows(output / "predictions.jsonl", predictions)

    metrics = {}
    for variant in ("baseline", "optimized"):
        metrics[variant] = _summary([row for key, row in all_rows.items() if key[0] == variant])
    before = metrics["baseline"]
    after = metrics["optimized"]
    improvements = {method: {
        "coverage_delta_pp": 100 * (after[method]["coverage_at_5"] - before[method]["coverage_at_5"]),
        "ndcg_delta": after[method]["ndcg_at_5"] - before[method]["ndcg_at_5"],
    } for method in METHODS}
    summary = dict(created_at=utc_now(), source="mal_recommendations", split="showcase",
                   n_targets=size, candidate_catalog_size=14202, k=5,
                   purpose="presentation-only high-consistency subset; not an unbiased test estimate",
                   selection=dict(rule=target_rows[0]["selection_rule"], size=size,
                                  source_metrics_sha256=sha256(optimized_metrics_path),
                                  source_truth_sha256=sha256(ROOT / "data/mal_ground_truth.jsonl"),
                                  source_test_targets=234, posthoc_truth_selection=True,
                                  model_selection_used=False),
                   before=before, after=after, improvements=improvements,
                   target_ids=[row["target_id"] for row in target_rows])
    write_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "showcase_targets", "before_coverage", "after_coverage",
                         "coverage_delta_pp", "before_ndcg", "after_ndcg", "ndcg_delta"])
        for method in METHODS:
            writer.writerow([method, size, before[method]["coverage_at_5"], after[method]["coverage_at_5"],
                             improvements[method]["coverage_delta_pp"], before[method]["ndcg_at_5"],
                             after[method]["ndcg_at_5"], improvements[method]["ndcg_delta"]])
    _plot(output, before, after, size)
    lines = ["# 算法①高一致性展示子集", "",
             f"本页选取 {size} 部目标动漫，仅用于课堂展示和逐例说明，不替代完整的 234 部 MAL 测试集。", "",
             "## 选择规则", "",
             "从已冻结、已评测的 234 部 MAL 测试目标中，按四种方法的 `coverage@5 + NDCG@5` 平均值降序取前 40 部；并列按 MAL ID 升序。这个规则使用了保存的真值和预测结果，因此是事后展示筛选，不能用于声称总体效果、模型选择或正式替换。", "",
             "## 展示结果", "",
             "| 方法 | 原配方覆盖率@5 | 优化后覆盖率@5 | 原配方 NDCG@5 | 优化后 NDCG@5 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | {before[method]['coverage_at_5']:.2%} | {after[method]['coverage_at_5']:.2%} | {before[method]['ndcg_at_5']:.4f} | {after[method]['ndcg_at_5']:.4f} |")
    lines += ["", "这个子集的数值较高，是因为目标按照已观察到的模型与 MAL 推荐一致程度筛选；它适合展示推荐案例，不代表随机目标或全目录表现。完整总体结果仍见上级 [REPORT.md](../REPORT.md)。", "",
              "![展示子集对比](comparison.png)", "", "目标 ID 与筛选排名见实验目录中的 `data/showcase_targets.jsonl`。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Showcase written: n={size}, D optimized coverage={after['D']['coverage_at_5']:.2%}, NDCG={after['D']['ndcg_at_5']:.4f}", flush=True)


def _plot(output, before, after, size):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = np.arange(len(METHODS))
    for axis, metric, title in zip(axes, ("coverage_at_5", "ndcg_at_5"), ("Coverage @5", "NDCG @5")):
        first = [before[m][metric] for m in METHODS]
        second = [after[m][metric] for m in METHODS]
        bars_a = axis.bar(x - .19, first, .36, label="Original", color="#9cb2cb")
        bars_b = axis.bar(x + .19, second, .36, label="Optimized", color="#226d96")
        for bars, values in ((bars_a, first), (bars_b, second)):
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=8, padding=3)
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, max(.05, max(first + second) * 1.3))
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    fig.suptitle(f"Algorithm 1 | showcase subset n={size}")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    svg = output / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=SHOWCASE_SIZE)
    args = parser.parse_args()
    if args.size < 20 or args.size > 100:
        parser.error("showcase size must be between 20 and 100")
    build(args.size)
