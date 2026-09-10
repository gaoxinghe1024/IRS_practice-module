# -*- coding: utf-8 -*-
"""
AnimeFinder 智能动漫推荐与发现系统 —— FastAPI 主应用。

四算法 × 界面对位（一算法一文件，见 services/）：
    ① knowledge.py       -> GET /api/anime/{id}/similar     详情页基于知识的相似推荐
    ② hybrid.py          -> GET /api/folders/{fid}/recommend 收藏页 Rule+Knowledge 混合推理
    ③ icf_ga_fusion.py   -> GET /api/recommend/home          主页核心推荐 ICF+GA 融合
                            GET /api/recommend/new           新剧速递（数据库直读，无算法）
    ④ rag.py             -> POST /api/chat                   RAG 聊天 / 名场面回忆
    组件：cf.py（算法②ICF）/ ga.py（算法③GA）/ watchplan.py（GA 附加应用）
    认证：auth.py -> /api/auth/*
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db, get_conn
from services import knowledge, cf, rag, watchplan, auth, icf_ga_fusion, hybrid

app = FastAPI(title="AnimeFinder API", version="1.1")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.on_event("startup")
def _startup():
    init_db()


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="minutes")


def _resolve_uid(token: str = "", user_id: int = 0) -> int:
    """鉴权辅助：带 token 走真实登录身份；无 token 退回 user_id（演示模式）。"""
    if token:
        u = auth.user_from_token(token)
        if not u:
            raise HTTPException(401, "登录已过期，请重新登录")
        return u["id"]
    return user_id or 1


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
# 推荐（四算法对位：主页=③ICF+GA；新剧=数据库直读）
# ---------------------------------------------------------------------------
@app.get("/api/recommend/home")
def recommend_home(user_id: int = 1, top_k: int = 5):
    """主页核心推荐（算法③融合）：算法② ICF 召回 Top-20 → 算法③ GA 重排 Top-K。"""
    return icf_ga_fusion.home_recommend(user_id, top_k=top_k)


@app.get("/api/recommend/new")
def recommend_new(limit: int = 6):
    """新剧速递：数据库直读（按 aired 年份倒序 + 热度），不经过任何推荐算法。"""
    conn = get_conn()
    rows = [_row_to_dict(r) for r in conn.execute(
        "SELECT * FROM anime ORDER BY CAST(aired AS INTEGER) DESC, popularity DESC LIMIT ?",
        (limit,)).fetchall()]
    conn.close()
    return {"items": rows, "algorithm": "数据库直读（无推荐算法）"}


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
# 认证：注册 / 登录 / 会话
# ---------------------------------------------------------------------------
class RegisterBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/register")
def auth_register(body: RegisterBody):
    try:
        return auth.register(body.username, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/auth/login")
def auth_login(body: RegisterBody):
    try:
        return auth.login(body.username, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))


class TokenBody(BaseModel):
    token: str = ""


@app.post("/api/auth/logout")
def auth_logout(body: TokenBody):
    auth.logout(body.token)
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(token: str = ""):
    u = auth.user_from_token(token)
    if not u:
        raise HTTPException(401, "登录已过期，请重新登录")
    return u


# ---------------------------------------------------------------------------
# 收藏夹体系
# ---------------------------------------------------------------------------
@app.get("/api/user/{user_id}/folders")
def list_folders(user_id: int):
    """收藏夹列表（含默认收藏夹 id=0），附条目数与一张封面。"""
    conn = get_conn()

    def _stats(fid: int) -> dict:
        n = conn.execute("SELECT COUNT(*) FROM favorites WHERE user_id=? AND folder_id=?",
                         (user_id, fid)).fetchone()[0]
        cov = conn.execute(
            """SELECT a.image_url FROM favorites f JOIN anime a ON a.id=f.anime_id
               WHERE f.user_id=? AND f.folder_id=? AND a.image_url<>'' LIMIT 1""",
            (user_id, fid)).fetchone()
        return {"count": n, "cover": cov["image_url"] if cov else ""}

    default_folder = {"id": 0, "name": "默认收藏夹", "created_at": ""}
    default_folder.update(_stats(0))
    folders = [default_folder]
    for r in conn.execute("SELECT id, name, created_at FROM fav_folders WHERE user_id=? ORDER BY id",
                          (user_id,)).fetchall():
        d = dict(r)
        d.update(_stats(r["id"]))
        folders.append(d)
    conn.close()
    return {"items": folders}


class FolderBody(BaseModel):
    token: str = ""
    user_id: int = 0
    name: str


@app.post("/api/folders")
def create_folder(body: FolderBody):
    uid = _resolve_uid(body.token, body.user_id)
    name = body.name.strip()[:30]
    if not name:
        raise HTTPException(400, "收藏夹名称不能为空")
    conn = get_conn()
    try:
        cur = conn.execute("INSERT INTO fav_folders (user_id, name, created_at) VALUES (?,?,?)",
                           (uid, name, _now()))
        conn.commit()
        fid = cur.lastrowid
    finally:
        conn.close()
    return {"ok": True, "id": fid, "name": name}


@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: int, token: str = "", user_id: int = 0):
    if folder_id == 0:
        raise HTTPException(400, "默认收藏夹不可删除")
    uid = _resolve_uid(token, user_id)
    conn = get_conn()
    row = conn.execute("SELECT id FROM fav_folders WHERE id=? AND user_id=?", (folder_id, uid)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "收藏夹不存在")
    conn.execute("UPDATE favorites SET folder_id=0 WHERE folder_id=?", (folder_id,))  # 条目移回默认夹
    conn.execute("DELETE FROM fav_folders WHERE id=?", (folder_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/folders/{folder_id}")
def folder_detail(folder_id: int, user_id: int = 0, token: str = ""):
    """收藏夹详情：夹信息 + 内含作品。folder_id=0 为默认收藏夹（按用户隔离）。"""
    uid = _resolve_uid(token, user_id)
    conn = get_conn()
    if folder_id == 0:
        folder = {"id": 0, "name": "默认收藏夹", "created_at": ""}
    else:
        row = conn.execute("SELECT id, name, created_at FROM fav_folders WHERE id=? AND user_id=?",
                           (folder_id, uid)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(404, "收藏夹不存在")
        folder = dict(row)
        conn = get_conn()
    items = [_row_to_dict(r) for r in conn.execute(
        """SELECT a.* FROM favorites f JOIN anime a ON a.id=f.anime_id
           WHERE f.folder_id=? AND f.user_id=? ORDER BY f.id DESC""", (folder_id, uid)).fetchall()]
    conn.close()
    return {"folder": folder, "items": items}


@app.get("/api/folders/{folder_id}/recommend")
def folder_recommend_api(folder_id: int, user_id: int = 0, token: str = "",
                         keywords: str = "", min_score: float = 0,
                         max_episodes: int = 0, limit: int = 8):
    """收藏页混合推理（算法②）：规则筛选 × 收藏夹知识画像。"""
    uid = _resolve_uid(token, user_id)
    return hybrid.favorites_fusion(folder_id, uid, keywords, min_score, max_episodes, limit)


# ---------------------------------------------------------------------------
# 交互：评分 / 收藏 / 个人数据
# ---------------------------------------------------------------------------
class RateBody(BaseModel):
    user_id: int = 0
    token: str = ""
    anime_id: int
    rating: float


class FavBody(BaseModel):
    user_id: int = 0
    token: str = ""
    anime_id: int
    folder_id: int = 0


@app.post("/api/rate")
def rate(body: RateBody):
    """真实评分：写入 ratings（带时间戳），同步用户均分，重建 ICF 矩阵。"""
    uid = _resolve_uid(body.token, body.user_id)
    if not (0 < body.rating <= 10):
        raise HTTPException(400, "评分需在 0-10 之间")
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO ratings (user_id,anime_id,rating,created_at) VALUES (?,?,?,?)",
                 (uid, body.anime_id, body.rating, _now()))
    conn.execute("UPDATE users SET mean_score=(SELECT AVG(rating) FROM ratings WHERE user_id=?) WHERE id=?",
                 (uid, uid))
    conn.commit()
    conn.close()
    cf.invalidate_cache()  # 新评分进入后重建 ICF 矩阵
    return {"ok": True}


@app.post("/api/favorite")
def favorite(body: FavBody):
    """收藏开关：对指定收藏夹（默认 0）切换；取消时从所有收藏夹移除。"""
    uid = _resolve_uid(body.token, body.user_id)
    conn = get_conn()
    exist = conn.execute("SELECT id, folder_id FROM favorites WHERE user_id=? AND anime_id=?",
                         (uid, body.anime_id)).fetchone()
    if exist:
        conn.execute("DELETE FROM favorites WHERE id=?", (exist["id"],))
        conn.commit()
        conn.close()
        return {"ok": True, "favorited": False}
    conn.execute("INSERT INTO favorites (user_id,anime_id,folder_id) VALUES (?,?,?)",
                 (uid, body.anime_id, body.folder_id))
    conn.commit()
    conn.close()
    return {"ok": True, "favorited": True}


class FavRemoveBody(BaseModel):
    user_id: int = 0
    token: str = ""
    anime_id: int
    folder_id: int = 0


@app.post("/api/favorites/remove")
def favorite_remove(body: FavRemoveBody):
    """从指定收藏夹移除单个条目（不动其他收藏夹）。"""
    uid = _resolve_uid(body.token, body.user_id)
    conn = get_conn()
    conn.execute("DELETE FROM favorites WHERE user_id=? AND anime_id=? AND folder_id=?",
                 (uid, body.anime_id, body.folder_id))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/tags")
def all_tags():
    """全库 genre + theme 标签（供前端筛选下拉框使用，标签应选择而非输入）。"""
    conn = get_conn()
    genres: set[str] = set()
    themes: set[str] = set()
    for g, t in conn.execute("SELECT genres, themes FROM anime").fetchall():
        genres.update(x.strip() for x in (g or "").split(",") if x.strip())
        themes.update(x.strip() for x in (t or "").split(",") if x.strip())
    conn.close()
    return {"genres": sorted(genres), "themes": sorted(themes)}


class FavMoveBody(BaseModel):
    user_id: int = 0
    token: str = ""
    anime_id: int
    from_folder: int = 0
    to_folder: int = 0


@app.post("/api/favorites/move")
def favorites_move(body: FavMoveBody):
    """把作品在收藏夹之间移动（夹内 → 夹外/另一夹）。"""
    uid = _resolve_uid(body.token, body.user_id)
    if body.from_folder == body.to_folder:
        return {"ok": True, "moved": False}
    conn = get_conn()
    cur = conn.execute(
        "UPDATE favorites SET folder_id=? WHERE user_id=? AND anime_id=? AND folder_id=?",
        (body.to_folder, uid, body.anime_id, body.from_folder))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "该作品不在来源收藏夹中")
    return {"ok": True, "moved": True}


@app.get("/api/user/{user_id}/favorites")
def user_favorites(user_id: int):
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.*, f.folder_id FROM favorites f JOIN anime a ON a.id=f.anime_id
           WHERE f.user_id=?""", (user_id,)).fetchall()
    conn.close()
    return {"items": [_row_to_dict(r) for r in rows]}


@app.get("/api/user/{user_id}/ratings")
def user_ratings(user_id: int):
    """评分历史（真实评分系统）：含时间、封面信息，按时间倒序。"""
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.id, a.title, a.title_cn, a.image_url, a.an_type, a.episodes, a.score,
                  r.rating, r.created_at
           FROM ratings r JOIN anime a ON a.id=r.anime_id
           WHERE r.user_id=? ORDER BY r.created_at DESC, r.id DESC LIMIT 100""",
        (user_id,)).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows]}


@app.get("/api/user/{user_id}/profile")
def user_profile(user_id: int):
    """个人详情页：用户信息 + 统计 + 偏好 genre Top5。"""
    conn = get_conn()
    u = conn.execute("SELECT id, username, gender, mean_score, created_at FROM users WHERE id=?",
                     (user_id,)).fetchone()
    if not u:
        conn.close()
        raise HTTPException(404, "user not found")
    n_ratings, avg = conn.execute(
        "SELECT COUNT(*), AVG(rating) FROM ratings WHERE user_id=?", (user_id,)).fetchone()
    n_favs = conn.execute("SELECT COUNT(*) FROM favorites WHERE user_id=?", (user_id,)).fetchone()[0]
    n_folders = conn.execute("SELECT COUNT(*) FROM fav_folders WHERE user_id=?", (user_id,)).fetchone()[0]
    genre_rows = conn.execute(
        """SELECT a.genres FROM ratings r JOIN anime a ON a.id=r.anime_id
           WHERE r.user_id=? AND r.rating>=8""", (user_id,)).fetchall()
    conn.close()

    genre_cnt: dict[str, int] = {}
    for r in genre_rows:
        for g in r["genres"].split(","):
            g = g.strip()
            if g:
                genre_cnt[g] = genre_cnt.get(g, 0) + 1
    top_genres = sorted(genre_cnt, key=genre_cnt.get, reverse=True)[:5]

    return {"user": dict(u),
            "stats": {"ratings": n_ratings, "avg_rating": round(avg, 2) if avg else 0,
                      "favorites": n_favs, "folders": n_folders},
            "top_genres": top_genres}


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
