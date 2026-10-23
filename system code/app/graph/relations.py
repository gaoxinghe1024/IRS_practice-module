"""Relation graph queries over the SQLite relations table (no graph database needed).

Only confirmed edges with a traceable source are used for facts; "prequel" is
the only relation treated as a viewing prerequisite, and only in release order:
a prequel that began airing after the title it precedes (Jujutsu Kaisen 0 for
Jujutsu Kaisen, No Game No Life: Zero) is reported as a later-released prequel,
not as something to watch first. Traversal is bounded and cycle-safe. A missing
relation does not prove a title has no prerequisite, and nothing is inferred
from title numbering ("Season 2").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Anime, Relation

MAX_DEPTH = 12


@dataclass
class Edge:
    src: str
    dst: str
    relation_type: str
    source_ref: str


def confirmed_edges(db: Session, relation_types: tuple[str, ...] | None = None) -> list[Edge]:
    q = select(Relation.from_item, Relation.to_item, Relation.relation_type, Relation.source_ref).where(
        Relation.to_item.is_not(None), Relation.status == "confirmed")
    if relation_types:
        q = q.where(Relation.relation_type.in_(relation_types))
    return [Edge(*r) for r in db.execute(q)]


_YEARS: dict[str, dict[str, float]] = {}


def start_years(snap) -> dict[str, float]:
    """item_key -> first-aired year of the served snapshot (cached per dataset version)."""
    out = _YEARS.get(snap.dataset_version)
    if out is None:
        out = _YEARS[snap.dataset_version] = {str(k): float(y) for k, y in zip(snap.keys, snap.year, strict=True)
                                              if math.isfinite(y)}
    return out


class PrequelGraph:
    """prequel_of[x] = titles that must come before x (confirmed 'prequel' edges, in release order).

    With `start_year`, a prequel that began airing in a later year than the title it precedes is kept out of the
    prerequisites (and listed by `later_prequels`); with an unknown year either way the edge is kept.
    """

    def __init__(self, edges: list[Edge], start_year: dict[str, float] | None = None):
        self.start_year = start_year or {}
        self.prequels: dict[str, list[Edge]] = {}
        for e in edges:
            if e.relation_type == "prequel":
                self.prequels.setdefault(e.src, []).append(e)
            elif e.relation_type == "sequel":
                # "A has sequel B" also means "B has prequel A"
                self.prequels.setdefault(e.dst, []).append(Edge(e.dst, e.src, "prequel", e.source_ref))
        for k in self.prequels:
            seen, uniq = set(), []
            for e in sorted(self.prequels[k], key=lambda x: x.dst):
                if e.dst not in seen:
                    seen.add(e.dst)
                    uniq.append(e)
            self.prequels[k] = uniq

    def _released_later(self, prequel: str, title: str) -> bool:
        a, b = self.start_year.get(prequel), self.start_year.get(title)
        return a is not None and b is not None and a > b

    def later_prequels(self, item: str) -> list[str]:
        """Direct prequels of `item` that began airing after it (story order, not viewing prerequisites)."""
        return [e.dst for e in self.prequels.get(item, []) if self._released_later(e.dst, item)]

    def chain(self, item: str) -> tuple[list[str], bool, list[str]]:
        """All transitive prequels of `item` in watch order (earliest first), cycle flag, source refs."""
        order: list[str] = []
        refs: list[str] = []
        cycle = False
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(x: str, depth: int) -> None:
            nonlocal cycle
            if depth > MAX_DEPTH:
                return
            for e in self.prequels.get(x, []):
                if self._released_later(e.dst, x):
                    continue
                if e.dst in visiting:
                    cycle = True
                    continue
                if e.dst in done:
                    continue
                visiting.add(e.dst)
                visit(e.dst, depth + 1)
                visiting.discard(e.dst)
                done.add(e.dst)
                order.append(e.dst)
                refs.append(e.source_ref)

        visiting.add(item)
        visit(item, 0)
        return order, cycle, sorted(set(refs))


def watch_order(db: Session, item_key: str, start_year: dict[str, float] | None = None) -> dict:
    edges = confirmed_edges(db, ("prequel", "sequel"))
    g = PrequelGraph(edges, start_year)
    before, cycle, refs = g.chain(item_key)
    later = g.later_prequels(item_key)
    sequels = sorted({e.dst for e in edges if e.src == item_key and e.relation_type == "sequel"})
    titles = dict(db.execute(select(Anime.item_key, Anime.title).where(
        Anime.item_key.in_(before + sequels + later + [item_key]))).all())
    return {"watch_first": [{"item_key": k, "title": titles.get(k)} for k in before],
            "direct_sequels": [{"item_key": k, "title": titles.get(k)} for k in sequels],
            "prequels_released_later": [{"item_key": k, "title": titles.get(k)} for k in later],
            "cycle_detected": cycle, "sources": refs,
            "note": None if (before or sequels) else
            "No confirmed prequel/sequel relations recorded for this title (this does not prove there are none)."}
