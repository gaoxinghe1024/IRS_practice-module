# -*- coding: utf-8 -*-
"""
模块三：Knowledge Discovery & Data Mining
Item-based Collaborative Filtering（ICF）候选召回。

当前实现：内存中的 item-item 余弦相似度（评分中心化）。
真实数据接入后，750 万行评分建议：
    1) 分块读取（chunksize）累加共现统计；
    2) 离线预计算 item-item 相似度矩阵存盘（joblib/parquet），
       本模块的对外函数签名保持不变。
"""
from database import get_conn

_sim_cache: dict[int, list[tuple[int, float]]] | None = None


def _load_matrix():
    """读入评分表，构建 item -> {user: rating(中心化)} 的稀疏结构。"""
    global _sim_cache
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT r.user_id, r.anime_id, r.rating, a.score AS global_score
        FROM ratings r JOIN anime a ON a.id = r.anime_id
        """
    ).fetchall()
    conn.close()

    # item -> {user: centered_rating}
    item_users: dict[int, dict[int, float]] = {}
    for r in rows:
        centered = r["rating"] - (r["global_score"] or 7.0)
        item_users.setdefault(r["anime_id"], {})[r["user_id"]] = centered
    return item_users


def _build_sim():
    """计算 item-item 余弦相似度，每个 item 保留 Top-50 相似项。"""
    global _sim_cache
    if _sim_cache is not None:
        return _sim_cache
    item_users = _load_matrix()
    ids = list(item_users.keys())
    sims: dict[int, list[tuple[int, float]]] = {i: [] for i in ids}
    for x in range(len(ids)):
        i = ids[x]
        ui = item_users[i]
        norm_i = sum(v * v for v in ui.values()) ** 0.5
        for y in range(x + 1, len(ids)):
            j = ids[y]
            uj = item_users[j]
            common = ui.keys() & uj.keys()
            if len(common) < 3:  # 至少 3 个共同评分用户，抑制噪声
                continue
            dot = sum(ui[u] * uj[u] for u in common)
            norm_j = sum(v * v for v in uj.values()) ** 0.5
            if norm_i == 0 or norm_j == 0:
                continue
            sim = dot / (norm_i * norm_j)
            if sim > 0:
                sims[i].append((j, sim))
                sims[j].append((i, sim))
    for i in sims:
        sims[i].sort(key=lambda t: t[1], reverse=True)
        sims[i] = sims[i][:50]
    _sim_cache = sims
    return sims


def invalidate_cache():
    """用户新增评分后调用，让相似度矩阵在下次请求时重建。"""
    global _sim_cache
    _sim_cache = None


def cf_candidates(user_id: int, top_n: int = 20) -> list[dict]:
    """根据用户已评分作品做 ICF 候选召回，返回 Top-N 候选（含召回理由）。"""
    sims = _build_sim()
    conn = get_conn()
    user_ratings = conn.execute(
        "SELECT anime_id, rating FROM ratings WHERE user_id=? AND rating>=8", (user_id,)
    ).fetchall()
    all_anime = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM anime").fetchall()}
    watched = {r["anime_id"] for r in conn.execute(
        "SELECT anime_id FROM ratings WHERE user_id=?", (user_id,))}
    conn.close()

    if not user_ratings:
        # 冷启动：回退到全局高分
        fallback = [a for aid, a in all_anime.items() if aid not in watched]
        fallback.sort(key=lambda a: a["score"], reverse=True)
        return fallback[:top_n]

    scores: dict[int, float] = {}
    reasons: dict[int, list[str]] = {}
    for r in user_ratings:
        aid, rating = r["anime_id"], r["rating"]
        weight = (rating - 7.0) if rating > 7 else 0.1
        for j, sim in sims.get(aid, [])[:20]:
            if j in watched:
                continue
            scores[j] = scores.get(j, 0.0) + sim * weight
            reasons.setdefault(j, []).append(all_anime[aid]["title_cn"] or all_anime[aid]["title"])

    ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:top_n]
    return [{**all_anime[aid], "cf_score": round(s, 4),
             "cf_reason": f"因为你喜欢《{reasons[aid][0]}》"}
            for aid, s in ranked]
