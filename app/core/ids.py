"""Item identifiers.

``mal:<id>``              a real MyAnimeList title (mal_id stored as an integer, unique)
``synthetic:<seed>:<n>``  a fictional title generated with <seed>; mal_id is NULL

Offline Kaggle user ids and application user ids are separate namespaces and
are never joined by value or by username.
"""

from __future__ import annotations

import re

_MAL = re.compile(r"^mal:(\d+)$")
_SYN = re.compile(r"^synthetic:(\d+):(\d{3,})$")


def mal_key(mal_id: int) -> str:
    return f"mal:{int(mal_id)}"


def synthetic_key(seed: int, n: int) -> str:
    return f"synthetic:{int(seed)}:{int(n):03d}"


def is_valid_item_key(key: str) -> bool:
    return bool(_MAL.match(key or "") or _SYN.match(key or ""))


def mal_id_of(key: str) -> int | None:
    m = _MAL.match(key or "")
    return int(m.group(1)) if m else None


def key_namespace(key: str) -> str:
    return (key or "").split(":", 1)[0]
