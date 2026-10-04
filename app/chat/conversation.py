"""Per-session constraint state for multi-turn chat.

A follow-up such as "at most 12 episodes" refines the current request (the
reference title and earlier conditions stay); a new, substantive request with
no refinement cue starts a fresh constraint state. Explicit numbers in the new
message override earlier values; a tag moves between include/exclude when the
user changes their mind. Session state is never written to the account
preferences unless the user explicitly asks to save it.
"""

from __future__ import annotations

import copy
import re

from app.chat.intent import ParsedQuery

DEFAULT_STATE = {"include_tags": [], "exclude_tags": [], "soft_tags": [], "soft_avoid_tags": [],
                 "max_episodes": None, "min_episodes": None, "min_score": None, "max_score": None,
                 "year_from": None, "year_to": None, "types": None,
                 "reference": None, "soft_preferences": [], "last_intent": None, "last_items": [], "turns": 0}
REFINEMENT_CUES = re.compile(r"^\s*(?:also|and|but|only|instead|now|make it|what about|how about|actually|"
                             r"then|ok(?:ay)?|same but|更|再|只要|改成|那|还有|另外)\b|^\s*(?:at most|at least|"
                             r"under|no more than|up to|without|no |not |shorter|longer|less|fewer|more)", re.I)


def _dedupe(pairs: list) -> list:
    out = []
    for p in pairs:
        if list(p) not in out:
            out.append(list(p))
    return out


def is_refinement(pq: ParsedQuery, message: str, state: dict) -> bool:
    if not state or state.get("turns", 0) == 0:
        return False
    if pq.reference_titles:
        return False
    words = len(pq.free_text.split())
    return bool(REFINEMENT_CUES.search(message)) or words <= 3


def merge(state: dict | None, pq: ParsedQuery, message: str) -> tuple[dict, list[str], bool]:
    prev = copy.deepcopy(state) if state else copy.deepcopy(DEFAULT_STATE)
    refinement = is_refinement(pq, message, prev)
    st = prev if (refinement and not pq.reset) else copy.deepcopy(DEFAULT_STATE)
    st["turns"] = prev.get("turns", 0)
    changes: list[str] = []
    if pq.clear_years and (st.get("year_from") is not None or st.get("year_to") is not None):
        changes.append("no year limit")
        st["year_from"] = st["year_to"] = None
    if (pq.year_from is None) != (pq.year_to is None) and st.get("year_from") is not None \
            and st.get("year_from") == st.get("year_to"):
        st["year_from"] = st["year_to"] = None     # "from 2010" then "2005 or later": an open limit replaces the year
    for key in ("max_episodes", "min_episodes", "min_score", "max_score", "year_from", "year_to", "types"):
        new = getattr(pq, key)
        if new not in (None, []) and new != st.get(key):
            changes.append(f"{key}: {st.get(key)} -> {new}")
            st[key] = new
    for t in pq.include_tags:
        if t in st["exclude_tags"]:
            st["exclude_tags"].remove(t)
            changes.append(f"no longer excluding {t[1]}")
        st["include_tags"].append(t)
    for t in pq.exclude_tags:
        if t in st["include_tags"]:
            st["include_tags"].remove(t)
            changes.append(f"no longer requiring {t[1]}")
        st["exclude_tags"].append(t)
    st["soft_tags"] += pq.soft_tags
    st["soft_avoid_tags"] += pq.soft_avoid_tags
    for k in ("include_tags", "exclude_tags", "soft_tags", "soft_avoid_tags"):
        st[k] = _dedupe(st[k])
    if pq.reference_titles:
        st["reference"] = pq.reference_titles[0]
    for sp in pq.soft_preferences:
        if sp not in st["soft_preferences"]:
            st["soft_preferences"].append(sp)
    if st["reference"] is None:
        st["soft_preferences"] = [s for s in st["soft_preferences"] if "reference" not in s]
    st["turns"] += 1
    return st, changes, refinement


def effective_intent(pq: ParsedQuery, st: dict, refinement: bool) -> str:
    if refinement and st.get("last_intent") in ("similar_to", "preference_recommendation", "viewing_plan") \
            and pq.intent in ("preference_recommendation", "anime_fact"):
        return st["last_intent"]
    if pq.intent == "preference_recommendation" and st.get("reference") and \
            "similar_to_reference" in st.get("soft_preferences", []):
        return "similar_to"
    return pq.intent


def constraint_chips(st: dict) -> list[dict]:
    chips = []
    if st.get("reference"):
        chips.append({"kind": "soft", "label": f"Like {st['reference']['title']}"})
    for _ns, v in st.get("include_tags", []):
        chips.append({"kind": "hard", "label": f"{v}"})
    for _ns, v in st.get("exclude_tags", []):
        chips.append({"kind": "hard", "label": f"No {v}"})
    for _ns, v in st.get("soft_tags", []):
        chips.append({"kind": "soft", "label": f"Preferably {v}"})
    for _ns, v in st.get("soft_avoid_tags", []):
        chips.append({"kind": "soft", "label": f"Not too much {v}"})
    if st.get("max_episodes") is not None:
        chips.append({"kind": "hard", "label": f"<= {st['max_episodes']} episodes"})
    if st.get("min_episodes") is not None:
        chips.append({"kind": "hard", "label": f">= {st['min_episodes']} episodes"})
    if st.get("min_score") is not None and st.get("min_score") == st.get("max_score"):
        chips.append({"kind": "hard", "label": f"Score = {st['min_score']}"})
    else:
        if st.get("min_score") is not None:
            chips.append({"kind": "hard", "label": f"Score >= {st['min_score']}"})
        if st.get("max_score") is not None:
            chips.append({"kind": "hard", "label": f"Score <= {st['max_score']}"})
    yf, yt = st.get("year_from"), st.get("year_to")
    if yf is not None and yt is not None:
        chips.append({"kind": "hard", "label": f"Began airing {yf}" if yf == yt else f"Began airing {yf}-{yt}"})
    elif yf is not None:
        chips.append({"kind": "hard", "label": f"Began airing {yf} or later"})
    elif yt is not None:
        chips.append({"kind": "hard", "label": f"Began airing {yt} or earlier"})
    if st.get("types"):
        chips.append({"kind": "hard", "label": "Type: " + "/".join(st["types"])})
    labels = {"shorter_than_reference": "Shorter than the reference", "short": "Preferably short",
              "long": "Preferably long", "longer_than_reference": "Longer than the reference",
              "obscure": "Less-known titles", "popular": "Popular titles", "high_score": "Highly rated"}
    for sp in st.get("soft_preferences", []):
        if sp in labels:
            chips.append({"kind": "soft", "label": labels[sp]})
    return chips
