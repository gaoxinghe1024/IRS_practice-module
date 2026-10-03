# -*- coding: utf-8 -*-
"""judge.py — LLM-judge harness: prompt construction, blind shuffling, aggregation.

Judges score listwise: per question, the 13 candidate answers (T + 6 models x R/N)
are anonymized as A1..A13 with a per-judge x per-question shuffle
(random.Random(f"42|{judge_label}|{qid}"), seed 42). Trap questions and
infeasible constraint questions additionally carry a boolean `frank` dimension.

DEFAULT IS REPLAY: the saved judge outputs (results/judge_raw.jsonl.gz) are
aggregated with the saved label maps (results/extra/judge_label_maps.jsonl.gz)
into per-answer scores (same-vendor judges excluded for the judge_avg caliber).
Live mode (--backend openai) needs LLM_API_* env vars and is not exercised by
the packaged results.

CLI:
    python harness/judge.py --aggregate            # replay aggregation (offline)
    python harness/judge.py --build-prompts        # rebuild the 240 judge prompts
    python harness/judge.py --build-prompts --live # then actually call a backend
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "algo"))
sys.path.insert(0, str(PKG / "harness"))

JUDGES = [("hy3", "Hy3"), ("deepseek-v4.1-flash", "Deepseek-V4.1-Flash"),
          ("glm-5.3", "GLM-5.3")]
JVENDOR = {"hy3": "Tencent", "deepseek-v4.1-flash": "DeepSeek", "glm-5.3": "Zhipu"}
MVENDOR = {"Hy3": "Tencent", "Deepseek-V4.1-Flash": "DeepSeek", "GLM-5.3": "Zhipu",
           "GLM-5.3-Flash": "Zhipu", "MiniMax-M3": "MiniMax", "Kimi-K3": "Moonshot",
           "T(system)": "system"}
CLI2DISPLAY = {"hy3": "Hy3", "deepseek-v4.1-flash": "Deepseek-V4.1-Flash",
               "glm-5.3": "GLM-5.3", "glm-5.3-flash": "GLM-5.3-Flash",
               "minimax-m3": "MiniMax-M3", "kimi-k3-1": "Kimi-K3"}
DIMS = ["relevance", "faithfulness", "usefulness", "tone"]


def load_jsonl(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_prompt(q: dict, ctx_rows: list[dict], answers: list[dict],
                 labels: list[str], need_frank: bool) -> str:
    """Byte-compatible with the evaluation's judge prompt construction."""

    def fmt_row(r: dict) -> str:
        if "anime_title" in r:  # scenes table row
            return (f"名场面[{r['anime_title']} 第{r['episode']}集]: "
                    f"{r.get('description', '')}（keywords: {r.get('keywords', '')}）")
        return (f"《{r.get('title_cn') or r['title']}》({r['title']}) 类型:{r.get('an_type')} "
                f"集数:{r.get('episodes')} 评分:{r.get('score')} 制作:{r.get('studios')} "
                f"题材:{r.get('genres')} 主题:{r.get('themes')} "
                f"简介:{(r.get('synopsis') or '')[:120]}")

    kb = "\n".join(fmt_row(r) for r in ctx_rows) or "（无）"
    if need_frank:
        dims = ("1. relevance 相关性与准确性（1-5）\n"
                "2. faithfulness 忠于知识库、不编造（1-5）\n"
                "3. usefulness 推荐理由具体有用（1-5）\n"
                "4. tone 语气自然像聊天（1-5）\n"
                "5. frank 布尔值：是否坦率说明「查不到/不在库内/不相关」而没有编造")
        fmt = '{"relevance": 1-5, "faithfulness": 1-5, "usefulness": 1-5, ' \
              '"tone": 1-5, "frank": true/false, "reason": "一句话理由"}'
    else:
        dims = ("1. relevance 相关性与准确性（1-5）\n"
                "2. faithfulness 忠于知识库、不编造（1-5）\n"
                "3. usefulness 推荐理由具体有用（1-5）\n"
                "4. tone 语气自然像聊天（1-5）")
        fmt = '{"relevance": 1-5, "faithfulness": 1-5, "usefulness": 1-5, ' \
              '"tone": 1-5, "reason": "一句话理由"}'
    ans_block = "\n\n".join(
        f"【{lab}】\n{a['reply']}" for lab, a in zip(labels, answers))
    return (
        "你是动漫推荐系统的评测评委。下面是同一个用户问题的多个匿名回答"
        "（标签 A1…An 随机打乱，与任何系统、模型无关），请逐条独立打分。\n\n"
        f"【用户问题】\n{q['query']}\n\n"
        f"【出题人说明】\n{q['judge_note']}\n\n"
        f"【知识库相关条目】\n{kb}\n\n"
        f"【评分维度】\n{dims}\n\n"
        f"【待评回答】\n{ans_block}\n\n"
        "【输出要求】只输出一个 JSON 对象，禁止输出任何其他文字或代码块标记。\n"
        f"格式：{{\"A1\": {fmt}, \"A2\": {fmt}, ...}}\n"
        "每个标签都必须出现，分数为整数。")


def plan_questions(testset: list[dict], raw: list[dict]):
    """Yield (judge_id, judge_label, qid, shuffled_answers, labels, need_frank)."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        if r.get("status") == "ok" and r.get("reply"):
            by_q[r["qid"]].append(r)
    for jmid, jlabel in JUDGES:
        for qid in sorted(by_q):
            q = next(x for x in testset if x["id"] == qid)
            answers = sorted(by_q[qid], key=lambda a: a["call_id"])
            rng = random.Random(f"42|{jlabel}|{qid}")
            order = list(range(len(answers)))
            rng.shuffle(order)
            labels = [f"A{i + 1}" for i in range(len(answers))]
            shuffled = [answers[i] for i in order]
            need_frank = (q["type"] == "trap"
                          or (q["type"] == "constraint"
                              and not q.get("expect", {}).get("feasible", True)))
            yield jmid, jlabel, qid, shuffled, labels, need_frank


def load_label_maps(results: Path) -> dict:
    path = results / "extra" / "judge_label_maps.jsonl.gz"
    maps = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            maps[(d["judge"], d["qid"])] = d["labels"]
    return maps


def aggregate(results: Path) -> list[dict]:
    """Replay aggregation: judge_raw.jsonl.gz + label maps -> per-answer rows."""
    label_maps = load_label_maps(results)
    judge_rows = load_jsonl(results / "judge_raw.jsonl.gz")
    per_answer: dict[str, list[dict]] = defaultdict(list)
    for jr in judge_rows:
        if jr.get("status") not in ("ok", "partial"):
            continue
        lmap = {x["label"]: x for x in label_maps[(jr["judge"], jr["qid"])]}
        for lab, sc in (jr.get("scores") or {}).items():
            if lab in lmap:
                per_answer[lmap[lab]["call_id"]].append({**sc, "_judge": jr["judge"]})
    rows = []
    for cid, alls in sorted(per_answer.items()):
        # call_id format: <cli_id>|<group>|<qid>|<rep>
        ml = CLI2DISPLAY.get(cid.split("|")[0]) if "|" in cid else None
        vendor = MVENDOR.get(ml)
        elig = [s for s in alls
                if vendor in (None, "system") or JVENDOR.get(s["_judge"]) != vendor] or alls
        row = {"call_id": cid, "judge_n": len(elig)}
        for d in DIMS:
            row[f"judge_{d}"] = (sum(s[d] for s in elig) / len(elig)) if elig else None
            row[f"judge_all_{d}"] = (sum(s[d] for s in alls) / len(alls)) if alls else None
        if row["judge_relevance"] is not None:
            row["judge_avg"] = sum(row[f"judge_{d}"] for d in DIMS) / 4
        if row["judge_all_relevance"] is not None:
            row["judge_all_avg"] = sum(row[f"judge_all_{d}"] for d in DIMS) / 4
        fr = [bool(s["frank"]) for s in elig if "frank" in s]
        row["judge_frank"] = (sum(fr) > len(fr) / 2) if fr else None
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Judge prompt construction / aggregation.")
    ap.add_argument("--aggregate", action="store_true",
                    help="replay aggregation of saved judge outputs (offline)")
    ap.add_argument("--build-prompts", action="store_true",
                    help="rebuild judge prompts from raw replies")
    ap.add_argument("--live", action="store_true",
                    help="with --build-prompts: call the OpenAI-compat backend")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.aggregate or args.build_prompts):
        ap.error("choose --aggregate or --build-prompts")

    results = Path(args.results_dir)
    if not results.is_absolute():
        results = PKG / args.results_dir

    if args.aggregate:
        rows = aggregate(results)
        out = Path(args.out) if args.out else results / "judge_scores_rebuilt.csv"
        if not out.is_absolute():
            out = PKG / out
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        import csv
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"aggregated {len(rows)} answers -> {out}")
        print("JUDGE_AGGREGATE_OK")
        return 0

    testset = load_jsonl(PKG / "data" / "testset_v1.jsonl")
    raw = load_jsonl(results / "raw.jsonl.gz")
    ctx = {d["id"]: d["kb_entries"]
           for d in load_jsonl(results / "extra" / "judge_context.jsonl.gz")}
    out_dir = PKG / "results" / "judge_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for jmid, _jl, qid, shuffled, labels, need_frank in plan_questions(testset, raw):
        q = next(x for x in testset if x["id"] == qid)
        prompt = build_prompt(q, ctx.get(qid, []), shuffled, labels, need_frank)
        d = out_dir / jmid
        d.mkdir(exist_ok=True)
        (d / f"{qid}.txt").write_text(prompt, encoding="utf-8", newline="\n")
        n += 1
    print(f"built {n} judge prompts -> {out_dir}"
          + (" (live calls NOT implemented in replay package)" if not args.live else ""))
    print("JUDGE_BUILD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
