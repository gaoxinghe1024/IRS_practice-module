# -*- coding: utf-8 -*-
"""run_generation.py — T / R / N generation harness.

DEFAULT IS REPLAY MODE: re-emit saved outputs from results/raw.jsonl.gz without
any network or model call. Live backends require an explicit --backend flag:

    python harness/run_generation.py --group R --model Kimi-K3                 # replay
    python harness/run_generation.py --group R --model Kimi-K3 \
        --backend openai --out results/live_R_Kimi.jsonl                       # live (LLM_API_* env)
    python harness/run_generation.py --list-models

Groups:
  R  - retrieval-augmented: prompt = algo.prompts.build_r_prompt(snapshot.retrieve(q), q)
  N  - no retrieval:        prompt = algo.prompts.build_n_prompt(q)
  T  - template pipeline:   output = algo.prompts.t_answer(q)  (never calls an LLM)

In replay mode the saved prompt (results/extra/parity_prompts/<group>/<qid>.txt)
is hashed and the saved reply returned by algo.llm_backend.ReplayBackend, so a
mismatched prompt would surface as an error instead of silently returning the
wrong answer.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "algo"))
sys.path.insert(0, str(PKG / "harness"))

import prompts  # noqa: E402
from llm_backend import ReplayBackend, OpenAICompatBackend, CodeBuddyCliBackend  # noqa: E402

GROUPS = ("T", "R", "N")


def load_models() -> list[dict]:
    """Minimal reader for harness/models.yaml (two-key list, no PyYAML needed)."""
    text = (PKG / "harness" / "models.yaml").read_text(encoding="utf-8")
    out = []
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*-\s*display:\s*(.+)$", line)
        if m:
            cur = {"display": m.group(1).strip()}
            out.append(cur)
            continue
        m = re.match(r"\s*cli_id:\s*(.+)$", line)
        if m and cur is not None:
            cur["cli_id"] = m.group(1).strip()
    return out


def load_testset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_prompt(group: str, query: str):
    if group == "N":
        return prompts.build_n_prompt(query), None
    ret = prompts.retrieve(query)
    if group == "R":
        return prompts.build_r_prompt(ret, query), ret
    raise ValueError("T group has no prompt (template pipeline)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="T/R/N generation. Default: replay saved outputs (offline).")
    ap.add_argument("--group", choices=GROUPS, help="which group to run")
    ap.add_argument("--model", default=None, help="display name, see --list-models")
    ap.add_argument("--qids", default=None, help="comma-separated qids (default: all 80)")
    ap.add_argument("--backend", choices=["replay", "openai", "codebuddy-cli"],
                    default="replay",
                    help="replay (default, offline) | openai (live, env LLM_API_*) "
                         "| codebuddy-cli (provenance only, requires login)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="output jsonl path")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    models = load_models()
    if args.list_models:
        for m in models:
            print(f"{m['display']}\t{m['cli_id']}")
        return 0
    if not args.group:
        ap.error("--group is required (or use --list-models)")

    model = next((m for m in models if m["display"] == args.model), None) \
        if args.model else None
    cli_id = model["cli_id"] if model else args.model

    results = Path(args.results_dir)
    if not results.is_absolute():
        results = PKG / args.results_dir
    testset = load_testset(PKG / "data" / "testset_v1.jsonl")
    qids = args.qids.split(",") if args.qids else [q["id"] for q in testset]
    qmap = {q["id"]: q for q in testset}

    if args.group == "T":
        if args.backend != "replay":
            print("T group is the deterministic template pipeline; use replay mode")
            return 2
        records = []
        for qid in qids:
            t = prompts.t_answer(qmap[qid]["query"])
            records.append({"qid": qid, "group": "T", "model_label": "T(system)",
                            "status": "ok", "reply": t["reply"],
                            "scene_focus_title": (t.get("scene_focus") or {}).get("title")})
    else:
        if args.backend == "codebuddy-cli":
            raise NotImplementedError(
                "codebuddy-cli backend is provenance-only (needs WorkBuddy login); "
                "see algo/llm_backend.py")
        if args.backend == "replay":
            backend = ReplayBackend(results)
        else:
            backend = OpenAICompatBackend(model=cli_id)
        records = []
        for qid in qids:
            prompt, _ret = build_prompt(args.group, qmap[qid]["query"])
            rec = backend.generate(prompt, model=cli_id)
            records.append({"qid": qid, "group": args.group,
                            "model_label": args.model, "status":
                            "ok" if rec["error"] is None else "error",
                            "reply": rec["text"], "error": rec["error"],
                            "usage": rec["usage"], "latency_s": rec["latency_s"]})

    out = Path(args.out) if args.out else \
        (results / f"generation_{args.group}_{(args.model or 'all').replace(' ', '_')}.jsonl")
    if not out.is_absolute():
        out = PKG / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    print(f"{args.group} group: {n_ok}/{len(records)} ok -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
