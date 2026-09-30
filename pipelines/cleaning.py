"""Field parsers for the MAL 2023 snapshot. Unknown stays unknown (None / empty).

No value is imputed: an unknown episode count never becomes 12, an unknown
duration never becomes 24 minutes, a missing score never becomes 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNKNOWN_TOKENS = {"", "unknown", "n/a", "na", "nan", "none", "null", "not available", "?", "-"}
SYNOPSIS_PLACEHOLDERS = {
    "no description available for this anime.",
    "no synopsis information has been added to this title.",
    "no synopsis has been added for this series yet.",
}
_CORP_SUFFIX = re.compile(r"^(inc\.?|ltd\.?|co\.?|llc|corp\.?|corporation|limited|gmbh|s\.a\.?|plc|k\.k\.?)$",
                          re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_SEASONS = {"winter", "spring", "summer", "fall"}


def clean_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).replace("﻿", "").strip()
    return None if s.casefold() in UNKNOWN_TOKENS else s


def parse_int(value) -> int | None:
    s = clean_str(value)
    if s is None:
        return None
    s = s.replace(",", "")
    try:
        f = float(s)
    except ValueError:
        return None
    if f != f or f < 0 or f != int(f):  # NaN, negative or fractional
        return None
    return int(f)


def parse_float(value, lo: float | None = None, hi: float | None = None) -> float | None:
    s = clean_str(value)
    if s is None:
        return None
    try:
        f = float(s.replace(",", ""))
    except ValueError:
        return None
    if f != f:
        return None
    if (lo is not None and f < lo) or (hi is not None and f > hi):
        return None
    return f


def parse_multi(value) -> list[str]:
    """Split a comma separated multi-value cell, re-attaching corporate suffixes.

    "Yamamura Animation, Inc., NHK" -> ["Yamamura Animation, Inc.", "NHK"].
    """
    s = clean_str(value)
    if s is None:
        return []
    out: list[str] = []
    for piece in (p.strip() for p in s.split(",")):
        if not piece:
            continue
        if out and _CORP_SUFFIX.match(piece):
            out[-1] = f"{out[-1]}, {piece}"
            continue
        if piece.casefold() in UNKNOWN_TOKENS:
            continue
        out.append(piece)
    seen: set[str] = set()
    uniq = []
    for v in out:
        k = v.casefold()
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq


def parse_synopsis(value) -> str | None:
    s = clean_str(value)
    if s is None or s.casefold() in SYNOPSIS_PLACEHOLDERS:
        return None
    return s


@dataclass(frozen=True)
class Duration:
    minutes: float | None          # per episode (for single-episode titles: the runtime)
    per_episode_explicit: bool
    raw: str | None


_DUR = re.compile(r"(?:(\d+)\s*hr\.?)?\s*(?:(\d+)\s*min\.?)?\s*(?:(\d+)\s*sec\.?)?", re.IGNORECASE)


def parse_duration(value) -> Duration:
    raw = clean_str(value)
    if raw is None:
        return Duration(None, False, None)
    per_ep = "per ep" in raw.casefold()
    body = re.sub(r"per ep\.?", "", raw, flags=re.IGNORECASE).strip()
    m = _DUR.fullmatch(body)
    if not m or not any(m.groups()):
        return Duration(None, per_ep, raw)
    hours, mins, secs = (int(g) if g else 0 for g in m.groups())
    total = hours * 60 + mins + secs / 60.0
    return Duration(round(total, 3) if total > 0 else None, per_ep, raw)


def resolve_runtime(duration: Duration, episodes: int | None) -> tuple[float | None, float | None, str | None]:
    """Return (minutes_per_episode, total_minutes, basis).

    MAL's Duration is a per-episode figure ("24 min per ep"); single-episode
    titles show the runtime without "per ep". A multi-episode title whose
    duration lacks "per ep" is ambiguous and stays unknown.
    """
    if duration.minutes is None:
        return None, None, None
    if duration.per_episode_explicit:
        total = duration.minutes * episodes if episodes else None
        return duration.minutes, (round(total, 3) if total else None), "per_episode_average"
    if episodes == 1:
        return duration.minutes, duration.minutes, "total_runtime"
    return None, None, None


def _parse_date_piece(piece: str) -> str | None:
    piece = piece.strip().rstrip(",")
    if not piece or piece == "?":
        return None
    m = re.fullmatch(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),\s*(\d{4})", piece)
    if m and m.group(1).lower() in _MONTHS:
        return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{4})", piece)
    if m and m.group(1).lower() in _MONTHS:
        return f"{int(m.group(2)):04d}-{_MONTHS[m.group(1).lower()]:02d}"
    m = re.fullmatch(r"(\d{4})", piece)
    if m:
        return m.group(1)
    return None


def parse_aired(value) -> tuple[str | None, str | None]:
    """ISO dates with the precision available ("1998", "1998-04", "1998-04-03")."""
    raw = clean_str(value)
    if raw is None:
        return None, None
    parts = [p.strip() for p in raw.split(" to ")]
    start = _parse_date_piece(parts[0]) if parts else None
    end = _parse_date_piece(parts[1]) if len(parts) > 1 else None
    return start, end


def parse_premiered(value) -> tuple[str | None, int | None]:
    raw = clean_str(value)
    if raw is None:
        return None, None
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", raw)
    if not m or m.group(1).lower() not in _SEASONS:
        return None, None
    return m.group(1).lower(), int(m.group(2))


def year_of(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return int(iso[:4])
    except ValueError:
        return None


def is_explicit(genres: list[str], content_rating: str | None, exclude_genres: list[str],
                exclude_rating_prefixes: list[str]) -> bool:
    gset = {g.casefold() for g in genres}
    if any(g.casefold() in gset for g in exclude_genres):
        return True
    if content_rating:
        cr = content_rating.strip().casefold()
        return any(cr.startswith(p.casefold()) for p in exclude_rating_prefixes)
    return False
