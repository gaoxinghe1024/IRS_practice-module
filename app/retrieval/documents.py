"""Knowledge documents for retrieval, each with provenance and a locator.

kinds
  attributes   structured catalogue facts (short, never chunked)
  synopsis     official catalogue synopsis, chunked ~150-350 tokens with a small overlap
  relations    confirmed relation facts (resolved anime edges and named external entries)
  characters   character list with its source
  scene        one verified / unverified scene per document (never merged across titles)
  review       user review text (weak opinion source; never treated as official plot fact)

Evaluation questions, answer keys and labels are never written into the corpus.
Placeholder synopses were already removed during preprocessing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from app.core.text import sha256_text


@dataclass
class Doc:
    doc_id: str
    item_key: str
    kind: str
    title: str
    body: str
    source_type: str
    source_ref: str
    spoiler_level: str = "none"
    language: str = "en"
    source_url: str | None = None
    locator: dict = field(default_factory=dict)
    retrieved_at: str | None = None
    content_hash: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["content_hash"] = d["content_hash"] or sha256_text(self.body)
        return d

    @property
    def dense_text(self) -> str:
        return self.body if self.kind == "attributes" else f"{self.title}. {self.body}"


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def chunk_text(text: str, count_tokens: Callable[[str], int], min_tokens: int = 150, max_tokens: int = 350,
               overlap_tokens: int = 40) -> list[tuple[int, int, str]]:
    """Sentence-packing chunker. Returns (char_start, char_end, text). Short texts stay whole."""
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [(0, len(text), text)]
    sents, pos = [], 0
    for s in _SENT.split(text):
        start = text.find(s, pos)
        sents.append((start, start + len(s), s))
        pos = start + len(s)
    # split any single over-long sentence on words
    fine: list[tuple[int, int, str]] = []
    for st, en, s in sents:
        if count_tokens(s) <= max_tokens:
            fine.append((st, en, s))
            continue
        words = list(re.finditer(r"\S+", s))
        cur = []
        for w in words:
            cur.append(w)
            if count_tokens(" ".join(x.group() for x in cur)) >= max_tokens:
                fine.append((st + cur[0].start(), st + cur[-1].end(), s[cur[0].start():cur[-1].end()]))
                cur = []
        if cur:
            fine.append((st + cur[0].start(), st + cur[-1].end(), s[cur[0].start():cur[-1].end()]))
    chunks, i = [], 0
    while i < len(fine):
        j, toks = i, 0
        while j < len(fine) and (toks + count_tokens(fine[j][2]) <= max_tokens or j == i):
            toks += count_tokens(fine[j][2])
            j += 1
        st, en = fine[i][0], fine[j - 1][1]
        chunks.append((st, en, text[st:en]))
        if j >= len(fine):
            break
        # overlap: step back over trailing sentences up to overlap_tokens (always advance >= 1)
        back, k = 0, j - 1
        while k > i + 1 and back + count_tokens(fine[k][2]) <= overlap_tokens:
            back += count_tokens(fine[k][2])
            k -= 1
        i = max(i + 1, k + 1 if back else j)
    # merge a tiny trailing chunk into the previous one when it fits
    if len(chunks) >= 2 and count_tokens(chunks[-1][2]) < min_tokens // 3:
        st = chunks[-2][0]
        en = chunks[-1][1]
        if count_tokens(text[st:en]) <= max_tokens:
            chunks[-2:] = [(st, en, text[st:en])]
    return chunks


def _fmt_num(v) -> str | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def attribute_doc_text(row: dict, tags: dict[str, list[str]]) -> str:
    parts = [f"{row['title']}"]
    if row.get("title_english") and row["title_english"] != row["title"]:
        parts[0] += f" (English title: {row['title_english']})"
    parts[0] += "."
    facts = [("Type", row.get("type")), ("Episodes", _fmt_num(row.get("episodes"))),
             ("Duration", row.get("duration_raw")), ("Status", row.get("status")),
             ("Aired", row.get("aired_raw")), ("Source", row.get("source_material")),
             ("Genres", ", ".join(tags.get("genre", [])) or None),
             ("Themes", ", ".join(tags.get("theme", [])) or None),
             ("Demographic", ", ".join(tags.get("demographic", [])) or None),
             ("Studios", ", ".join(tags.get("studio", [])) or None),
             ("Catalogue score (MAL 2023 snapshot)" if row["item_key"].startswith("mal:") else "Catalogue score",
              _fmt_num(row.get("score"))), ("Content rating", row.get("content_rating"))]
    for label, val in facts:
        parts.append(f"{label}: {val if val not in (None, '') else 'unknown'}.")
    return " ".join(parts)


def _source_ref(row: dict) -> str:
    prov = row.get("provenance") or {}
    if row["item_key"].startswith("mal:"):
        return f"{prov.get('source', 'kaggle')}#anime_id={row['mal_id']}"
    return f"synthetic:seed{prov.get('seed')}#{row['item_key']}"


def build_documents(catalog: pd.DataFrame, tags: pd.DataFrame, relations: pd.DataFrame, characters: pd.DataFrame,
                    scenes: pd.DataFrame, reviews: list[dict] | None, count_tokens: Callable[[str], int],
                    cfg: dict) -> list[Doc]:
    served = catalog[catalog["in_catalog"].astype(bool)]
    tag_map: dict[str, dict[str, list[str]]] = {}
    for key, ns, val in tags[["item_key", "namespace", "value"]].itertuples(index=False):
        tag_map.setdefault(key, {}).setdefault(ns, []).append(val)
    title_of = dict(zip(catalog["item_key"], catalog["title"], strict=True))
    docs: list[Doc] = []
    mn, mx, ov = (int(cfg.get("chunk_min_tokens", 150)), int(cfg.get("chunk_max_tokens", 350)),
                  int(cfg.get("chunk_overlap_tokens", 40)))
    for row in served.to_dict("records"):
        key = row["item_key"]
        ref = _source_ref(row)
        docs.append(Doc(f"{key}#attr", key, "attributes", row["title"], attribute_doc_text(row, tag_map.get(key, {})),
                        "catalog_structured", ref, locator={"fields": "catalogue row"}))
        syn = row.get("synopsis")
        if isinstance(syn, str) and syn.strip():
            for n, (st, en, chunk) in enumerate(chunk_text(syn, count_tokens, mn, mx, ov)):
                docs.append(Doc(f"{key}#syn-{n}", key, "synopsis", row["title"], chunk, "catalog_synopsis", ref,
                                spoiler_level="minor",
                                locator={"parent": f"{key}#synopsis", "chunk": n, "char_start": st, "char_end": en}))
    served_keys = set(served["item_key"])
    if len(relations):
        for key, grp in relations.groupby("from_item"):
            if key not in served_keys:
                continue
            lines = []
            for r in grp.to_dict("records"):
                rt = str(r["relation_type"]).replace("_", " ")
                if r["to_entity_type"] == "anime" and r.get("to_item") in served_keys:
                    lines.append(f"{title_of[key]} has {rt}: {title_of[r['to_item']]}.")
                elif r.get("to_name"):
                    lines.append(f"{title_of[key]} has {rt} ({r['to_entity_type']}): {r['to_name']}.")
            if lines:
                refs = sorted(set(grp["source_ref"].astype(str)))
                docs.append(Doc(f"{key}#rel", key, "relations", title_of[key], " ".join(lines), "relation_facts",
                                "; ".join(refs), locator={"relations": len(lines)}))
    if len(characters):
        for key, grp in characters.groupby("item_key"):
            if key not in served_keys:
                continue
            mains = [n for n, role in zip(grp["name"], grp["role"], strict=True) if str(role).lower() == "main"]
            others = [n for n, role in zip(grp["name"], grp["role"], strict=True) if str(role).lower() != "main"]
            body = f"Characters of {title_of[key]}. Main: {', '.join(mains) or 'unknown'}."
            if others:
                body += f" Supporting: {', '.join(others[:12])}."
            docs.append(Doc(f"{key}#chars", key, "characters", title_of[key], body, "character_list",
                            str(grp["source_ref"].iloc[0])))
    if scenes is not None and len(scenes):
        for s in scenes.to_dict("records"):
            if s["item_key"] not in served_keys:
                continue
            loc = {"scene_id": s["scene_id"], "episode": s.get("episode"), "timestamp_seconds": s.get("timestamp_seconds")}
            loc = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in loc.items()}
            docs.append(Doc(f"scene:{s['scene_id']}", s["item_key"], "scene", title_of[s["item_key"]], s["description"],
                            "scene_verified" if s.get("verification_status") == "verified" else "scene_unverified",
                            s["source_ref"], spoiler_level=s.get("spoiler_level") or "major",
                            source_url=s.get("source_url"), locator=loc))
    for rv in reviews or []:
        key = rv.get("item_key")
        if key in served_keys and rv.get("text"):
            for n, (st, en, chunk) in enumerate(chunk_text(rv["text"], count_tokens, mn, mx, ov)):
                docs.append(Doc(f"review:{rv['review_id']}#{n}", key, "review", title_of[key], chunk,
                                "user_review_weak", rv.get("source_ref", "reviews.jsonl"), spoiler_level="major",
                                locator={"review_id": rv["review_id"], "chunk": n}))
    return docs
