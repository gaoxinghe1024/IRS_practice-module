# -*- coding: utf-8 -*-
"""R and N prompt construction, byte-identical to the prompts used in the evaluation.

R: the string the system's `_llm_reply` builds as its single user message, i.e. the
system line, the knowledge-base context from `_build_context` and the user query.
N: the role line and the query only, no knowledge base.
T: `rag_snapshot.chat(query)` with the LLM variables cleared; no prompt, the template
reply is produced directly.

tests/test_parity.py checks all of them against the 80 saved prompts.
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
