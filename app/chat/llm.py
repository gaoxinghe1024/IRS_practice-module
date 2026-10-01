"""Configurable LLM adapters (provider, base URL, model, timeout, token budget from config/env).

openai_compatible  plain HTTPS chat/completions (DeepSeek, OpenAI, vLLM, Ollama, LM Studio, ...)
anthropic          Claude through the official ``anthropic`` SDK (optional ``llm`` extra):
                   JSON-schema structured output (output_config.format), stop_reason
                   "refusal" handled before reading content, and server-side refusal
                   fallback enabled (beta server-side-fallback-2026-07-01, fallbacks="default").
                   No sampling parameters are sent (current Claude models reject them).

The key is read only from ANILOGIC_LLM_API_KEY. A missing key or provider never
blocks the rest of the system: callers fall back to the local evidence templates.
Every outcome is explicit (ok / timeout / rate_limited / invalid_json / empty /
refusal / http_error / connection_error); usage-based cost is ``None`` (unknown)
whenever usage or prices are missing, never silently 0.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import Settings

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class LLMResult:
    ok: bool
    provider: str
    model: str
    data: dict | None = None
    text: str | None = None
    error: str | None = None
    detail: str | None = None
    stop_reason: str | None = None
    usage: dict | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"ok": self.ok, "provider": self.provider, "model": self.model, "error": self.error,
                "stop_reason": self.stop_reason, "usage": self.usage, "cost_usd": self.cost_usd,
                "cost_status": "measured" if self.cost_usd is not None else "unknown",
                "latency_ms": round(self.latency_ms, 1), **({"extra": self.extra} if self.extra else {})}


def _cost(settings: Settings, usage: dict | None) -> float | None:
    pin, pout = settings.llm.price_per_mtok_input, settings.llm.price_per_mtok_output
    if not usage or pin is None or pout is None:
        return None
    if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    return round(usage["input_tokens"] / 1e6 * pin + usage["output_tokens"] / 1e6 * pout, 6)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str | None) -> dict | None:
    if not text or not text.strip():
        return None
    t = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


class OpenAICompatibleClient:
    provider = "openai_compatible"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        cfg = settings.llm
        self.model = cfg.model
        self.base_url = cfg.base_url.rstrip("/")
        self.client = httpx.Client(timeout=cfg.timeout_s, transport=transport)

    def _post(self, body: dict) -> httpx.Response:
        return self.client.post(f"{self.base_url}/chat/completions", json=body,
                                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"})

    def complete(self, system: str, user: str, json_schema: dict | None = None,
                 max_tokens: int | None = None) -> LLMResult:
        cfg = self.settings.llm
        body: dict[str, Any] = {"model": self.model, "temperature": cfg.temperature,
                                "max_tokens": int(max_tokens or cfg.max_output_tokens),
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]}
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        res = LLMResult(ok=False, provider=self.provider, model=self.model)
        try:
            r = self._post(body)
            if r.status_code == 400 and "response_format" in r.text and "response_format" in body:
                body.pop("response_format")          # provider without JSON mode: prompt-only JSON
                r = self._post(body)
        except httpx.TimeoutException:
            res.error = "timeout"
        except httpx.HTTPError as exc:
            res.error, res.detail = "connection_error", type(exc).__name__
        else:
            if r.status_code == 429:
                res.error, res.detail = "rate_limited", f"retry-after={r.headers.get('retry-after')}"
            elif r.status_code >= 400:
                res.error, res.detail = "http_error", f"status={r.status_code}"
            else:
                try:
                    payload = r.json()
                    choice = (payload.get("choices") or [{}])[0]
                    res.text = (choice.get("message") or {}).get("content")
                    res.stop_reason = choice.get("finish_reason")
                    u = payload.get("usage") or {}
                    res.usage = ({"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}
                                 if u else None)
                except ValueError:
                    res.error = "invalid_json"
                if res.error is None:
                    if not res.text or not res.text.strip():
                        res.error = "empty"
                    elif json_schema is not None:
                        res.data = parse_json_object(res.text)
                        res.error = None if res.data is not None else "invalid_json"
                    res.ok = res.error is None
        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.cost_usd = _cost(self.settings, res.usage)
        return res


class AnthropicClient:
    provider = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: Settings, client=None):
        import anthropic

        self.settings = settings
        self.model = settings.llm.model or ANTHROPIC_DEFAULT_MODEL
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic(api_key=settings.llm_api_key, timeout=settings.llm.timeout_s,
                                                    max_retries=1)

    def complete(self, system: str, user: str, json_schema: dict | None = None,
                 max_tokens: int | None = None) -> LLMResult:
        a = self._anthropic
        cfg = self.settings.llm
        output_config: dict[str, Any] = {"effort": getattr(cfg, "effort", None) or "low"}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        t0 = time.perf_counter()
        res = LLMResult(ok=False, provider=self.provider, model=self.model)
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=int(max_tokens or getattr(cfg, "anthropic_max_tokens", 16000)),
                system=system, messages=[{"role": "user", "content": user}], output_config=output_config,
                betas=[self.FALLBACK_BETA], fallbacks="default")
        except a.RateLimitError as exc:
            res.error, res.detail = "rate_limited", f"retry-after={exc.response.headers.get('retry-after')}"
        except a.APITimeoutError:
            res.error = "timeout"
        except a.APIStatusError as exc:
            res.error, res.detail = "http_error", f"status={exc.status_code}"
        except a.APIConnectionError:
            res.error = "connection_error"
        else:
            res.stop_reason = resp.stop_reason
            usage = getattr(resp, "usage", None)
            if usage is not None:
                res.usage = {"input_tokens": getattr(usage, "input_tokens", None),
                             "output_tokens": getattr(usage, "output_tokens", None)}
                iterations = getattr(usage, "iterations", None) or []
                if any(getattr(it, "type", None) == "fallback_message" for it in iterations):
                    res.extra["served_by_fallback"] = getattr(resp, "model", None)
            if resp.stop_reason == "refusal":
                details = getattr(resp, "stop_details", None)
                res.error, res.detail = "refusal", getattr(details, "category", None)
            else:
                res.text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
                if not res.text:
                    res.error = "empty"
                elif json_schema is not None:
                    res.data = parse_json_object(res.text)
                    res.error = None if res.data is not None else "invalid_json"
                res.ok = res.error is None
        res.latency_ms = (time.perf_counter() - t0) * 1000
        # a fallback model bills at its own rates -> the configured price no longer applies
        res.cost_usd = None if res.extra.get("served_by_fallback") else _cost(self.settings, res.usage)
        return res


def get_llm(settings: Settings):
    provider = settings.llm.provider
    if provider == "none" or not settings.llm_api_key:
        return None
    if provider == "openai_compatible":
        if not settings.llm.base_url or not settings.llm.model:
            return None
        return OpenAICompatibleClient(settings)
    if provider == "anthropic":
        try:
            return AnthropicClient(settings)
        except ImportError:
            return None
    return None
