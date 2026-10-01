"""Lexical retrieval over SQLite FTS5 (BM25) with eligible-item restriction.

Hard constraints are applied BEFORE ranking by restricting the FTS query to the
eligible item keys (json_each), so "top-5 of the whole catalogue, then filter to
nothing" cannot happen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

STOPWORDS = set("""a an the and or but if then than of to in on at by for with about into over under from up down out
is are was were be been being do does did have has had i me my we our you your he she it its they them their this
that these those what which who whom where when why how any some no not only very can could should would will just
anime show series something like similar want wanted looking watch watching find recommend recommendation one
please give me more less short long shorter longer episode episodes ep eps""".split())

TOKEN = re.compile(r"[\w']+", re.UNICODE)


def fts_terms(query: str, keep_stopwords: bool = False) -> list[str]:
    toks = [t.lower().strip("'") for t in TOKEN.findall(query or "")]
    out = []
    for t in toks:
        if not t or (not keep_stopwords and t in STOPWORDS) or len(t) < 2:
            continue
        if t not in out:
            out.append(t)
    return out[:24]


def fts_query(terms: list[str]) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


@dataclass
class LexicalHit:
    doc_id: str
    item_key: str
    kind: str
    score: float      # bm25 (lower is better in SQLite; we negate so higher is better)
    rank: int


class LexicalRetriever:
    def __init__(self, engine: Engine):
        self.engine = engine

    def search(self, query: str, eligible_keys: list[str] | None, top_n: int = 50,
               kinds: list[str] | None = None) -> list[LexicalHit]:
        terms = fts_terms(query)
        if not terms:
            return []
        sql = ("SELECT doc_id, item_key, kind, bm25(docs_fts, 0.0, 0.0, 0.0, 2.0, 1.0) AS s FROM docs_fts "
               "WHERE docs_fts MATCH :q")
        params: dict = {"q": fts_query(terms), "n": top_n}
        if eligible_keys is not None:
            sql += " AND item_key IN (SELECT value FROM json_each(:keys))"
            params["keys"] = json.dumps(list(eligible_keys))
        if kinds:
            sql += " AND kind IN (SELECT value FROM json_each(:kinds))"
            params["kinds"] = json.dumps(kinds)
        sql += " ORDER BY s, doc_id LIMIT :n"
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).all()
        return [LexicalHit(r[0], r[1], r[2], -float(r[3]), i + 1) for i, r in enumerate(rows)]

    def search_titles(self, query: str, top_n: int = 10) -> list[tuple[str, float]]:
        terms = fts_terms(query, keep_stopwords=True)
        if not terms:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT item_key, bm25(anime_fts, 0.0, 3.0, 1.0) AS s FROM anime_fts "
                                     "WHERE anime_fts MATCH :q ORDER BY s, item_key LIMIT :n"),
                                {"q": " AND ".join('"' + t.replace('"', '""') + '"*' for t in terms), "n": top_n}).all()
        return [(r[0], -float(r[1])) for r in rows]
