# -*- coding: utf-8 -*-
"""prompts.py — R / N prompt construction, byte-identical to the evaluation.

R prompt = the exact string the system's `_llm_reply` builds (single user message):
    SYSTEM_R + "\\n\\n知识库：\\n" + _build_context(retrieved) + "\\n\\n用户：" + query
N prompt = role line only (no knowledge base, no KB-dependent wording).
T group  = rag_snapshot.chat(query) with LLM env vars explicitly cleared
           (no prompt file - the template reply is produced directly).

Parity against the 80 saved R/N prompts is checked by tests/test_parity.py.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rag_snapshot  # noqa: E402

SYSTEM_R = ("你是一个懂动漫的推荐助手，请根据下面的知识库内容用中文自然地回答用户，"
            "不要编造知识库里没有的作品。")
SYSTEM_N = "你是一个懂动漫的推荐助手，请用中文自然地回答用户。"


def build_r_prompt(retrieved: dict, message: str) -> str:
    """R-group prompt: verbatim reconstruction of the system's LLM request body."""
    context = rag_snapshot._build_context(retrieved)
    return SYSTEM_R + "\n\n知识库：\n" + context + "\n\n用户：" + message


def build_n_prompt(message: str) -> str:
    """N-group prompt: role line only."""
    return SYSTEM_N + "\n\n用户：" + message


def retrieve(message: str) -> dict:
    """Retrieval via the frozen snapshot (reads data/kb/anime.db)."""
    return rag_snapshot.retrieve(message)


def t_answer(message: str) -> dict:
    """T group: snapshot chat() with LLM env vars explicitly cleared (never calls an LLM)."""
    rag_snapshot.LLM_API_BASE = ""
    rag_snapshot.LLM_API_KEY = ""
    return rag_snapshot.chat(message)
