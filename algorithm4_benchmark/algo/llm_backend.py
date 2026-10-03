# -*- coding: utf-8 -*-
"""llm_backend.py — unified model backend interface for the evaluation kit.

    generate(prompt, **opts) -> {"text": str|None, "usage": dict|None,
                                 "latency_s": float|None, "error": str|None}

Three implementations:

* ReplayBackend       - returns saved outputs from results/ (indexed by prompt hash);
                        zero network, fully offline. This is the DEFAULT for all
                        tests and replay analysis.
* OpenAICompatBackend - live OpenAI-compatible HTTP backend (env: LLM_API_BASE,
                        LLM_API_KEY, LLM_MODEL). Request shape mirrors the system's
                        `_llm_reply` (see algo/rag_snapshot.py): a single user message,
                        temperature 0.6, 20 s timeout, POST {base}/chat/completions.
* CodeBuddyCliBackend - provenance only. The ORIGINAL evaluation ran through the
                        CodeBuddy CLI (WorkBuddy built-in models, account login
                        required), so it cannot run elsewhere; kept to document how
                        the saved outputs were produced. generate() raises.

Honesty note (also in README): the live backends were only exercised against a
fake OpenAI-compatible server on 127.0.0.1 (see tests/test_backends.py). No real
vendor API was ever called from this package.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

TEMPERATURE = 0.6   # same as system _llm_reply
TIMEOUT_S = 20      # same as system _llm_reply


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class ReplayBackend:
    """Returns saved model outputs; indexes them by sha256(prompt).

    Index sources (both under <package>/results/):
      * extra/parity_prompts/{R,N}/{qid}.txt  - the 160 saved prompts
      * raw.jsonl.gz                          - the saved replies (group R/N)
    For T-group prompts (template replies, no prompt file) use lookup()
    with (group, qid) instead.
    """

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = Path(results_dir) if results_dir else _pkg_root() / "results"
        self._by_hash: dict[str, dict] = {}
        self._by_key: dict[tuple, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        raw = load_jsonl(self.results_dir / "raw.jsonl.gz")
        by_key = {}
        for r in raw:
            if r.get("group") in ("R", "N") and r.get("status") == "ok":
                by_key[(r["group"], r["qid"], r.get("model_label"))] = r
        pdir = self.results_dir / "extra" / "parity_prompts"
        for group in ("R", "N"):
            gdir = pdir / group
            if not gdir.is_dir():
                continue
            for pf in sorted(gdir.glob("*.txt")):
                prompt = pf.read_text(encoding="utf-8")
                qid = pf.stem
                for model_label in sorted({k[2] for k in by_key if k[0] == group and k[1] == qid}):
                    rec = by_key.get((group, qid, model_label))
                    if rec is None:
                        continue
                    entry = {"text": rec.get("reply") or "",
                             "usage": {"input_tokens": rec.get("input_tokens"),
                                       "output_tokens": rec.get("output_tokens")},
                             "latency_s": (rec.get("duration_ms") or 0) / 1000.0,
                             "error": None}
                    h = hashlib.sha256(
                        (prompt + "\x00" + (model_label or "")).encode("utf-8")).hexdigest()
                    self._by_hash.setdefault(h, entry)
                    self._by_key[(group, qid, model_label)] = entry

    def generate(self, prompt: str, model: str | None = None, **opts) -> dict:
        h = hashlib.sha256((prompt + "\x00" + (model or "")).encode("utf-8")).hexdigest()
        if h in self._by_hash:
            return dict(self._by_hash[h])
        return {"text": None, "usage": None, "latency_s": None,
                "error": "no saved output for this prompt/model pair"}

    def lookup(self, group: str, qid: str, model_label: str) -> dict:
        rec = self._by_key.get((group, qid, model_label))
        if rec is None:
            return {"text": None, "usage": None, "latency_s": None,
                    "error": f"no saved output for {group}/{qid}/{model_label}"}
        return dict(rec)


class OpenAICompatBackend:
    """Live OpenAI-compatible backend. Request shape mirrors the system _llm_reply."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout_s: float = TIMEOUT_S,
                 temperature: float = TEMPERATURE):
        self.base_url = (base_url or os.environ.get("LLM_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.timeout_s = timeout_s
        self.temperature = temperature

    def build_request(self, prompt: str) -> tuple[urllib.request.Request, bytes]:
        """Exposed for tests: build (Request, body) exactly as the system does."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        return req, body

    def generate(self, prompt: str, **opts) -> dict:
        t0 = time.perf_counter()
        if not (self.base_url and self.api_key and self.model):
            return {"text": None, "usage": None, "latency_s": None,
                    "error": "LLM_API_BASE / LLM_API_KEY / LLM_MODEL not configured"}
        try:
            timeout = float(opts.get("timeout_s", self.timeout_s))
            req, _ = self.build_request(prompt)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            latency = time.perf_counter() - t0
            return {"text": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage"),
                    "latency_s": latency,
                    "error": None}
        except urllib.error.HTTPError as e:
            return {"text": None, "usage": None,
                    "latency_s": time.perf_counter() - t0, "error": f"HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001 - mirrors the system's bare except
            return {"text": None, "usage": None,
                    "latency_s": time.perf_counter() - t0, "error": str(e)}


class CodeBuddyCliBackend:
    """Provenance documentation only - the channel that produced the saved outputs.

    The original evaluation invoked (per call):

        codebuddy.exe -p --model <cli-id> --output-format json --max-turns 1 \
            --no-session-persistence --tools '' --strict-mcp-config \
            --mcp-config '{"mcpServers":{}}' \
            --system-prompt 'You are a helpful assistant.' < prompt.txt > out.json

    This requires an interactive WorkBuddy / CodeBuddy login and is therefore not
    runnable from this package; generate() always raises.
    """

    CLI_FLAGS = ["-p", "--output-format", "json", "--max-turns", "1",
                 "--no-session-persistence", "--tools", "",
                 "--strict-mcp-config", "--system-prompt", "You are a helpful assistant."]

    def generate(self, prompt: str, model: str | None = None, **opts) -> dict:
        raise NotImplementedError(
            "CodeBuddyCliBackend is documentation-only: the original evaluation "
            "channel requires a WorkBuddy/CodeBuddy login and cannot run from "
            "this package. Use ReplayBackend for offline reproduction.")
