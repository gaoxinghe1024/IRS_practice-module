# -*- coding: utf-8 -*-
"""
算法①：Knowledge-Based Reasoning（基于知识的推理推荐）
=======================================================
对应界面：动漫详情页「相似推荐」（每个作品页的小推荐位）。

相似度公式（与项目文档一致）：
    Similarity(i,j) = w1*GenreSim + w2*ThemeSim + w3*DemographicSim
                    + w4*StudioSim + w5*LengthSim
附：轻量知识图谱（Anime/Character/Studio/Franchise 四类节点）。
"""
from database import get_conn

WEIGHTS = dict(genre=0.35, theme=0.3, demographic=0.15, studio=0.1, length=0.1)


def _to_set(s: str) -> set:
    return {x.strip() for x in (s or "").split(",") if x.strip()}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _length_sim(e1: int, e2: int) -> float:
    """集数接近视为相似，指数衰减。"""
    if e1 <= 0 or e2 <= 0:
        return 0.5
    diff = abs(e1 - e2)
    return 1.0 / (1.0 + diff / 12.0)


def anime_similarity(a: dict, b: dict) -> float:
    return (
        WEIGHTS["genre"] * _jaccard(_to_set(a["genres"]), _to_set(b["genres"]))
        + WEIGHTS["theme"] * _jaccard(_to_set(a["themes"]), _to_set(b["themes"]))
        + WEIGHTS["demographic"] * (1.0 if a["demographics"] == b["demographics"] and a["demographics"] else 0.0)
        + WEIGHTS["studio"] * (1.0 if a["studios"] == b["studios"] and a["studios"] else 0.0)
        + WEIGHTS["length"] * _length_sim(a["episodes"], b["episodes"])
    )


def similar_anime(anime_id: int, limit: int = 5, offset: int = 0) -> list[dict]:
    """详情页『换一换』相似推荐：offset 控制 1-5 / 6-10 / 11-15 名。"""
    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM anime").fetchall()]
    conn.close()
    target = next((r for r in rows if r["id"] == anime_id), None)
    if not target:
        return []
    scored = []
    for r in rows:
        if r["id"] == anime_id:
            continue
        sim = anime_similarity(target, r)
        scored.append({**r, "similarity": round(sim, 4),
                       "reason_tags": sorted(_to_set(target["genres"]) & _to_set(r["genres"]))
                       + sorted(_to_set(target["themes"]) & _to_set(r["themes"]))})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[offset: offset + limit]


# ---------------------------------------------------------------------------
# 轻量知识图谱：四类节点（Anime / Character / Studio / Franchise）
# 三类边（belongs_to / produced_by / features）。真实数据接入后可平滑升级 networkx。
# ---------------------------------------------------------------------------
def build_kg(anime_id: int) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT * FROM anime WHERE id=?", (anime_id,)).fetchone()
    conn.close()
    if not row:
        return {"nodes": [], "edges": []}
    a = dict(row)
    nodes = [{"id": f"anime-{a['id']}", "label": a["title_cn"] or a["title"], "type": "Anime"}]
    edges = []

    def add(node_type: str, label: str, edge_label: str):
        nid = f"{node_type.lower()}-{label}"
        if not any(n["id"] == nid for n in nodes):
            nodes.append({"id": nid, "label": label, "type": node_type})
        edges.append({"source": f"anime-{a['id']}", "target": nid, "label": edge_label})

    for c in _to_set(a["characters"]):
        add("Character", c, "features")
    for s in _to_set(a["studios"]):
        add("Studio", s, "produced_by")
    if a["franchise"]:
        add("Franchise", a["franchise"], "belongs_to")
        # 同系列作品
        conn = get_conn()
        sib = conn.execute("SELECT id,title,title_cn FROM anime WHERE franchise=? AND id<>?",
                           (a["franchise"], anime_id)).fetchall()
        conn.close()
        for s in sib:
            sid = f"anime-{s['id']}"
            if not any(n["id"] == sid for n in nodes):
                nodes.append({"id": sid, "label": s["title_cn"] or s["title"], "type": "Anime"})
            edges.append({"source": sid, "target": f"franchise-{a['franchise']}", "label": "belongs_to"})
    return {"nodes": nodes, "edges": edges}
