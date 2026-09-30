"""Text normalisation helpers shared by search, rules and parsing."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def norm(text: str) -> str:
    """NFKC + casefold + collapsed whitespace (for keyword and tag matching)."""
    return _WS.sub(" ", nfkc(text).casefold()).strip()


def norm_title(text: str) -> str:
    """Stronger normalisation for title / alias lookup (punctuation removed)."""
    return _WS.sub(" ", _PUNCT.sub(" ", nfkc(text).casefold())).strip()


def split_keywords(raw: str) -> list[str]:
    """Comma separated keywords (ASCII or full-width comma), normalised and de-duplicated."""
    parts = re.split(r"[,，、;；]", nfkc(raw or ""))
    out: list[str] = []
    for p in parts:
        k = norm(p)
        if k and k not in out:
            out.append(k)
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    """Rough token count (word + punctuation pieces) used when no model tokenizer is loaded."""
    return len(re.findall(r"\w+|[^\w\s]", text or "", re.UNICODE))


def is_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" for ch in text or "")
