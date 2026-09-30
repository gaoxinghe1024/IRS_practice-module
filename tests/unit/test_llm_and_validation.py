import json
from types import SimpleNamespace

import httpx

from app.chat.grounded_answer import run_llm, validate
from app.chat.llm import AnthropicClient, OpenAICompatibleClient, get_llm, parse_json_object
from app.core.config import load_settings


def _settings(**llm):
    env = {"ANILOGIC_LLM__PROVIDER": "openai_compatible", "ANILOGIC_LLM__BASE_URL": "https://llm.example/v1",
           "ANILOGIC_LLM__MODEL": "test-model", "ANILOGIC_LLM_API_KEY": "sk-test"}
    env.update({f"ANILOGIC_LLM__{k.upper()}": str(v) for k, v in llm.items()})
    return load_settings("synthetic", environ=env)


def test_get_llm_requires_provider_and_key():
    assert get_llm(load_settings("synthetic", environ={})) is None
    s = load_settings("synthetic", environ={"ANILOGIC_LLM__PROVIDER": "openai_compatible"})
    assert get_llm(s) is None          # no key -> local mode, never a crash


def _client(handler, **llm):
    return OpenAICompatibleClient(_settings(**llm), transport=httpx.MockTransport(handler))


def test_openai_compatible_success_usage_and_unknown_cost():
    def handler(req):
        body = json.loads(req.content)
        assert req.headers["authorization"] == "Bearer sk-test" and body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "hi"}'}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    res = _client(handler).complete("sys", "user", json_schema={"type": "object"})
    assert res.ok and res.data == {"answer": "hi"} and res.usage == {"input_tokens": 10, "output_tokens": 5}
    assert res.cost_usd is None                     # no configured price -> unknown, not 0
    priced = _client(handler, price_per_mtok_input=1.0, price_per_mtok_output=2.0).complete("s", "u", json_schema={})
    assert priced.cost_usd == round(10 / 1e6 + 5 * 2 / 1e6, 6)


def test_openai_compatible_failures_are_explicit():
    assert _client(lambda r: httpx.Response(429, headers={"retry-after": "7"})).complete("s", "u").error == "rate_limited"
    assert _client(lambda r: httpx.Response(500)).complete("s", "u").error == "http_error"
    bad = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}))
    assert bad.complete("s", "u", json_schema={}).error == "invalid_json"
    empty = _client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}))
    assert empty.complete("s", "u").error == "empty"

    def timeout(req):
        raise httpx.ReadTimeout("slow")
    assert _client(timeout).complete("s", "u").error == "timeout"
    calls = []

    def no_json_mode(req):
        calls.append(json.loads(req.content))
        if "response_format" in calls[-1]:
            return httpx.Response(400, text="unsupported response_format")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"a": 1}'}}]})
    assert _client(no_json_mode).complete("s", "u", json_schema={}).data == {"a": 1} and len(calls) == 2


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic(); records the request and returns a canned message."""

    def __init__(self, message):
        self.message = message
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.message


def _anthropic_settings():
    return load_settings("synthetic", environ={"ANILOGIC_LLM__PROVIDER": "anthropic", "ANILOGIC_LLM_API_KEY": "k"})


def test_anthropic_adapter_request_shape_and_json():
    msg = SimpleNamespace(stop_reason="end_turn", stop_details=None, model="claude-opus-5-5",
                          content=[SimpleNamespace(type="thinking", thinking=""),
                                   SimpleNamespace(type="text", text='{"answer": "x", "item_keys": []}')],
                          usage=SimpleNamespace(input_tokens=100, output_tokens=20, iterations=None))
    fake = _FakeAnthropic(msg)
    res = AnthropicClient(_anthropic_settings(), client=fake).complete("sys", "question", json_schema={"type": "object"})
    assert res.ok and res.data["answer"] == "x"
    k = fake.kwargs
    assert k["model"] == "claude-opus-5-5" and "temperature" not in k
    assert k["output_config"]["format"]["type"] == "json_schema"
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_refusal_is_not_read_as_content():
    msg = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"), model="m",
                          content=[], usage=SimpleNamespace(input_tokens=5, output_tokens=0, iterations=None))
    res = AnthropicClient(_anthropic_settings(), client=_FakeAnthropic(msg)).complete("s", "u", json_schema={})
    assert not res.ok and res.error == "refusal" and res.detail == "cyber"


FACTS = [{"item_key": "synthetic:1:001", "title": "A", "episodes": 12, "catalog_score": 7.5},
         {"item_key": "synthetic:1:002", "title": "B", "episodes": 24, "catalog_score": 8.1}]
DOCS = [{"doc_id": "synthetic:1:001#attr", "item_key": "synthetic:1:001", "kind": "attributes", "body": "A ...",
         "source_type": "catalog_structured", "locator": {}}]


def test_validation_drops_invented_items_refs_and_numbers():
    ans = {"answer": "Try A, it has 12 episodes.", "item_keys": ["synthetic:1:001", "synthetic:9:999"],
           "claims": [{"text": "A has 12 episodes", "item_key": "synthetic:1:001", "evidence_refs": ["synthetic:1:001#attr"]},
                      {"text": "A has 99 episodes", "item_key": "synthetic:1:001", "evidence_refs": ["synthetic:1:001#attr"]},
                      {"text": "A is great", "item_key": "synthetic:1:001", "evidence_refs": ["made-up-doc"]}],
           "unknowns": []}
    clean, v = validate(ans, {f["item_key"] for f in FACTS}, {d["doc_id"]: d for d in DOCS}, {f["item_key"]: f for f in FACTS})
    assert v.ok and clean["item_keys"] == ["synthetic:1:001"]
    assert [c["text"] for c in clean["claims"]] == ["A has 12 episodes"]
    assert "synthetic:9:999" in v.dropped_items and "made-up-doc" in v.dropped_refs


def test_unsupported_episode_number_rejects_answer_text():
    ans = {"answer": "That scene is in episode 7.", "item_keys": ["synthetic:1:001"], "claims": [], "unknowns": []}
    _, v = validate(ans, {"synthetic:1:001"}, {d["doc_id"]: d for d in DOCS}, {f["item_key"]: f for f in FACTS})
    assert not v.ok and v.text_rejected


def test_run_llm_one_repair_then_gives_up():
    class Stub:
        def __init__(self):
            self.n = 0

        def complete(self, system, user, json_schema=None, max_tokens=None):
            from app.chat.llm import LLMResult

            self.n += 1
            return LLMResult(ok=True, provider="stub", model="m", data={"answer": "Watch Z (99 episodes).",
                                                                        "item_keys": ["synthetic:7:777"],
                                                                        "claims": [], "unknowns": []})
    stub = Stub()
    clean, calls, v = run_llm(stub, "q", "preference_recommendation", {}, FACTS, DOCS, max_calls=2)
    assert clean is None and stub.n == 2 and len(calls) == 2 and not v.ok


def test_parse_json_object_handles_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("prefix {\"b\": 2} suffix") == {"b": 2}
    assert parse_json_object("[1, 2]") is None
