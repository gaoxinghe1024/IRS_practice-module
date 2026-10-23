# -*- coding: utf-8 -*-
"""test_backends.py — backend contract tests against a fake OpenAI-compatible
server on 127.0.0.1 (no real vendor API is ever contacted)."""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "algo"))

from llm_backend import (CodeBuddyCliBackend, OpenAICompatBackend,  # noqa: E402
                         ReplayBackend, TIMEOUT_S)


class _FakeHandler(BaseHTTPRequestHandler):
    last_body = {}
    last_headers = {}
    last_path = ""

    def do_POST(self):
        _FakeHandler.last_path = self.path
        _FakeHandler.last_headers = dict(self.headers)
        length = int(self.headers.get("Content-Length", 0))
        _FakeHandler.last_body = json.loads(self.rfile.read(length) or b"{}")
        if "/timeout" in self.path:
            import time
            time.sleep(2)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
            return
        if "/error" in self.path:
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error": "rate limited"}')
            return
        resp = {"choices": [{"message": {"content": "FAKE-OK"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


class TestBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_openai_request_shape(self):
        b = OpenAICompatBackend(base_url=f"http://127.0.0.1:{self.port}",
                                api_key="test", model="fake-model")
        out = b.generate("你好")
        self.assertIsNone(out["error"])
        self.assertEqual(out["text"], "FAKE-OK")
        self.assertEqual(out["usage"]["completion_tokens"], 5)
        self.assertGreater(out["latency_s"], 0)
        # request shape mirrors the system _llm_reply: single user message,
        # temperature 0.6, Authorization header
        self.assertEqual(_FakeHandler.last_path, "/chat/completions")
        self.assertEqual(_FakeHandler.last_body["messages"],
                         [{"role": "user", "content": "你好"}])
        self.assertEqual(_FakeHandler.last_body["temperature"], 0.6)
        self.assertEqual(_FakeHandler.last_body["model"], "fake-model")
        self.assertEqual(_FakeHandler.last_headers.get("Authorization"),
                         "Bearer test")

    def test_openai_http_error(self):
        b = OpenAICompatBackend(base_url=f"http://127.0.0.1:{self.port}/error",
                                api_key="k", model="m")
        out = b.generate("x")
        self.assertIsNone(out["text"])
        self.assertEqual(out["error"], "HTTP 429")

    def test_openai_timeout(self):
        b = OpenAICompatBackend(base_url=f"http://127.0.0.1:{self.port}/timeout",
                                api_key="k", model="m", timeout_s=0.3)
        out = b.generate("x")
        self.assertIsNone(out["text"])
        self.assertIn("timed out", (out["error"] or "").lower())

    def test_missing_config_is_error_not_crash(self):
        b = OpenAICompatBackend(base_url="", api_key="", model="")
        out = b.generate("x")
        self.assertIsNone(out["text"])
        self.assertIn("not configured", out["error"])

    def test_replay_backend_returns_saved_outputs(self):
        rb = ReplayBackend()
        # pick any saved R reply through lookup()
        rec = rb.lookup("R", "F03", "Kimi-K3")
        self.assertIsNone(rec["error"])
        self.assertTrue(rec["text"])
        miss = rb.generate("这是一个从未出现过的 prompt")
        self.assertIsNone(miss["text"])
        self.assertIn("no saved output", miss["error"])

    def test_codebuddy_backend_is_provenance_only(self):
        with self.assertRaises(NotImplementedError):
            CodeBuddyCliBackend().generate("x")
        self.assertIn("--max-turns", " ".join(CodeBuddyCliBackend.CLI_FLAGS))
        self.assertEqual(TIMEOUT_S, 20)


if __name__ == "__main__":
    unittest.main()
