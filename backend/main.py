# -*- coding: utf-8 -*-
"""
AnimeFinder 智能动漫推荐与发现系统 —— FastAPI 主应用。

技术对应：
    /api/anime/{id}/similar          -> Decision Automation / Knowledge-Based Reasoning
    /api/recommend/personal          -> 规则筛选 + 基于收藏的知识推理（混合架构）
    /api/recommend/home              -> ICF 召回 + GA 重排（数据挖掘 + 资源优化）
    /api/chat                        -> RAG + 自然语言交互（认知系统）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db, get_conn
from services import knowledge, cf, ga, rag, watchplan

app = FastAPI(title="AnimeFinder API", version="1.0")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.on_event("startup")
def _startup():
    init_db()


# ---------------------------------------------------------------------------
# 基础：列表 / 搜索 / 详情
# ---------------------------------------------------------------------------
def _row_to_dict(r) -> dict:
    d = dict(r)
    d["genres"] = [g for g in d["genres"].split(",") if g]
    d["themes"] = [t for t in d["themes"].split(",") if t]
    d["characters"] = [c for c in d["characters"].split(",") if c]
    return d


@app.get("/api/anime/list")
def anime_list(q: str = "", genre: str = "", page: int = 1, page_size: int = 12):
    conn = get_conn()
    rows = [_row_to_dict(r) for r in conn.execute("SELECT * FROM anime ORDER BY score DESC").fetchall()]
    conn.close()
    if q:
        key = q.lower()
        rows = [r for r in rows if key in r["title"].lower() or key in r["title_cn"]
                or any(key in c.lower() for c in r["characters"])]
    if genre:
        rows = [r for r in rows if genre in r["genres"]]
    total = len(rows)
    start = (page - 1) * page_size
    return {"total": total, "page": page, "items": rows[start: start + page_size]}


@app.get("/api/anime/{anime_id}")
def anime_detail(anime_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM anime WHERE id=?", (anime_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "anime not found")
    return _row_to_dict(row)


@app.get("/api/anime/{anime_id}/similar")
def anime_similar(anime_id: int, offset: int = 0, limit: int = Query(5, le=10)):
    """基于知识的相似推荐；offset 支持『换一换』（1-5名 / 6-10名 / ...）。"""
    return knowledge.similar_anime(anime_id, limit=limit, offset=offset)


@app.get("/api/anime/{anime_id}/kg")
def anime_kg(anime_id: int):
    """轻量知识图谱：Anime/Character/Studio/Franchise 四类节点。"""
    return knowledge.build_kg(anime_id)


# ---------------------------------------------------------------------------
# 推荐
# ---------------------------------------------------------------------------
@app.get("/api/recommend/home")
def recommend_home(user_id: int = 1, top_k: int = 5):
    """主页个性推荐：ICF 召回 Top-20 → GA 重排 Top-K。"""
    candidates = cf.cf_candidates(user_id, top_n=20)
    optimized, best_fit, baseline_fit = ga.optimize(candidates, user_id, top_k=top_k)
    return {
        "items": optimized,
        "ga": {"best_fitness": round(best_fit, 4), "baseline_fitness": round(baseline_fit, 4),
               "improved": round(best_fit - baseline_fit, 4),
               "weights": ga.W},
    }


@app.get("/api/recommend/personal")
def recommend_personal(user_id: int = 1, keywords: str = "", min_score: float = 0,
                       max_episodes: int = 0, limit: int = 8):
    """规则筛选 + 基于收藏的知识推荐（混合推理架构）。"""
    conn = get_conn()
    rows = [_row_to_dict(r) for r in conn.execute("SELECT * FROM anime").fetchall()]
    fav_rows = conn.execute(
        """SELECT a.genres, a.themes FROM favorites f JOIN anime a ON a.id=f.anime_id
           WHERE f.user_id=?""", (user_id,)).fetchall()
    conn.close()

    # 1) 规则层：显式硬约束过滤
    pool = rows
    if keywords:
        kws = [k.strip() for k in keywords.replace("，", ",").split(",") if k.strip()]
        pool = [r for r in pool if all(
            any(k in t for t in r["genres"] + r["themes"]) or k in r["title_cn"] or k in r["title"]
            for k in kws)]
    if min_score > 0:
        pool = [r for r in pool if r["score"] >= min_score]
    if max_episodes > 0:
        pool = [r for r in pool if not r["episodes"] or r["episodes"] <= max_episodes]

    # 2) 知识推理层：与收藏画像的标签匹配分排序
    fav_tags: dict[str, int] = {}
    for r in fav_rows:
        for t in r["genres"].split(",") + r["themes"].split(","):
            t = t.strip()
            if t:
                fav_tags[t] = fav_tags.get(t, 0) + 1
    for r in pool:
        match = sum(fav_tags.get(t, 0) for t in r["genres"] + r["themes"])
        r["match_score"] = round(r["score"] / 10 + match * 0.15, 3)
    pool.sort(key=lambda r: r["match_score"], reverse=True)
    return {"items": pool[:limit], "profile_tags": sorted(fav_tags, key=fav_tags.get, reverse=True)[:8]}


# ---------------------------------------------------------------------------
# 观看计划优化（GA 组合优化：时间预算约束）
# ---------------------------------------------------------------------------
class PlanBody(BaseModel):
    user_id: int = 1
    hours: float = 15
    keywords: str = ""


@app.post("/api/watchplan")
def watch_plan(body: PlanBody):
    """GA 组合优化：max Preference+Quality+Diversity s.t. 总时长 <= 预算。"""
    return watchplan.make_plan(body.user_id, body.hours, body.keywords)


# ---------------------------------------------------------------------------
# 交互：评分 / 收藏 / 我的收藏
# ---------------------------------------------------------------------------
class RateBody(BaseModel):
    user_id: int = 1
    anime_id: int
    rating: float


class FavBody(BaseModel):
    user_id: int = 1
    anime_id: int


@app.post("/api/rate")
def rate(body: RateBody):
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO ratings (user_id,anime_id,rating) VALUES (?,?,?)",
                 (body.user_id, body.anime_id, body.rating))
    conn.commit()
    conn.close()
    cf.invalidate_cache()  # 新评分进入后重建 ICF 矩阵
    return {"ok": True}


@app.post("/api/favorite")
def favorite(body: FavBody):
    conn = get_conn()
    exist = conn.execute("SELECT id FROM favorites WHERE user_id=? AND anime_id=?",
                         (body.user_id, body.anime_id)).fetchone()
    if exist:
        conn.execute("DELETE FROM favorites WHERE id=?", (exist["id"],))
        conn.commit()
        conn.close()
        return {"ok": True, "favorited": False}
    conn.execute("INSERT INTO favorites (user_id,anime_id) VALUES (?,?)", (body.user_id, body.anime_id))
    conn.commit()
    conn.close()
    return {"ok": True, "favorited": True}


@app.get("/api/user/{user_id}/favorites")
def user_favorites(user_id: int):
    conn = get_conn()
    rows = [_row_to_dict(r) for r in conn.execute(
        """SELECT a.* FROM favorites f JOIN anime a ON a.id=f.anime_id WHERE f.user_id=?""",
        (user_id,)).fetchall()]
    conn.close()
    return {"items": rows}


@app.get("/api/user/{user_id}/ratings")
def user_ratings(user_id: int):
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.id, a.title, a.title_cn, r.rating FROM ratings r
           JOIN anime a ON a.id=r.anime_id WHERE r.user_id=? ORDER BY r.id DESC LIMIT 30""",
        (user_id,)).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# 聊天机器人（RAG）
# ---------------------------------------------------------------------------
class ChatBody(BaseModel):
    message: str
    user_id: int = 1


@app.post("/api/chat")
def chat(body: ChatBody):
    return rag.chat(body.message)


# ---------------------------------------------------------------------------
# 静态前端
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ---------------------------------------------------------------------------
# 本地开发入口：在 PyCharm 中直接运行本文件即可启动服务
#   等价于命令行: uvicorn main:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

