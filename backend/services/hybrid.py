# -*- coding: utf-8 -*-
"""
算法②：Rule-Based + Knowledge-Based Hybrid Fusion（收藏页混合推理）
====================================================================
对应界面：用户收藏夹详情页「基于此收藏夹的混合推理推荐」。

两层结构（不涉及协同过滤 / 遗传算法，与主页 ICF+GA 严格区分）：
    规则层 Rule-Based     —— 显式硬约束筛选：关键词 / 最低评分 / 集数上限
    知识层 Knowledge-Based —— 用收藏夹内作品的 genre/theme 建立偏好画像，
                              对通过规则的候选做标签匹配打分排序
        match_score = MAL评分/10 + 标签命中数 × 0.15

若规则层把候选全部筛空，自动降级为「纯知识画像排序」并标记 relaxed=True。
"""
from database import get_conn
from services.knowledge import _to_set


def _parse_keywords(keywords: str) -> list[str]:
    return [k.strip() for k in (keywords or "").replace("，", ",").split(",") if k.strip()]


def _rule_pass(a: dict, kws: list[str], min_score: float, max_episodes: int) -> bool:
    """规则层：全部硬条件都满足才通过。"""
    if kws and not all(
        any(k in t for t in a["genres"] + a["themes"]) or k in a["title_cn"] or k in a["title"]
        for k in kws
    ):
        return False
    if min_score > 0 and a["score"] < min_score:
        return False
    if max_episodes > 0 and a["episodes"] and a["episodes"] > max_episodes:
        return False
    return True


def favorites_fusion(folder_id: int, user_id: int, keywords: str = "", min_score: float = 0,
                     max_episodes: int = 0, limit: int = 8) -> dict:
    """收藏页混合推理：规则筛选 × 收藏夹知识画像。"""
    conn = get_conn()
    # folder_id=0 是全用户共享的虚拟"默认夹"，必须按用户过滤
    folder_items = [dict(r) for r in conn.execute(
        """SELECT a.* FROM favorites f JOIN anime a ON a.id=f.anime_id
           WHERE f.folder_id=? AND f.user_id=?""", (folder_id, user_id)).fetchall()]
    fav_ids = {r["anime_id"] for r in conn.execute(
        "SELECT anime_id FROM favorites WHERE user_id=?", (user_id,)).fetchall()}
    all_anime = [{**dict(r), "genres": r["genres"].split(","),
                  "themes": r["themes"].split(",")} for r in
                 conn.execute("SELECT * FROM anime").fetchall()]
    conn.close()

    if not folder_items:
        return {"items": [], "profile_tags": [], "relaxed": False,
                "message": "收藏夹还是空的，先收藏几部作品，混合推理就有素材了"}

    # --- 知识层画像：收藏夹内 genre/theme 出现频次（folder_items 里是逗号分隔原始串） ---
    profile: dict[str, int] = {}
    for it in folder_items:
        for t in _to_set(it["genres"]) | _to_set(it["themes"]):
            profile[t] = profile.get(t, 0) + 1
    profile_tags = sorted(profile, key=profile.get, reverse=True)[:8]

    # --- 规则层 ---
    kws = _parse_keywords(keywords)
    passed = [a for a in all_anime if a["id"] not in fav_ids
              and _rule_pass(a, kws, min_score, max_episodes)]
    relaxed = False
    if not passed:  # 规则筛空 → 降级纯知识层
        relaxed = True
        passed = [a for a in all_anime if a["id"] not in fav_ids]

    # --- 知识层打分排序 ---
    for a in passed:
        match = sum(profile.get(t.strip(), 0) for t in a["genres"] + a["themes"] if t.strip())
        a["match_score"] = round(a["score"] / 10 + match * 0.15, 3)
        hits = sorted({t.strip() for t in a["genres"] + a["themes"] if profile.get(t.strip())})
        a["fusion_reason"] = ("画像命中：" + "、".join(hits[:4])) if hits else "综合质量优选"
    passed.sort(key=lambda x: x["match_score"], reverse=True)

    return {"items": passed[:limit], "profile_tags": profile_tags, "relaxed": relaxed,
            "rules_applied": {"keywords": kws, "min_score": min_score, "max_episodes": max_episodes}}
