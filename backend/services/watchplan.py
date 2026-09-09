# -*- coding: utf-8 -*-
"""
附加功能 6.3：动漫观看计划优化（GA 组合优化）

对应问题定义：用户时间预算有限（如"这个周末有15个小时"）时，
需要的不是若干孤立高分作品，而是一份兼顾 总分/偏好/多样性/时长约束
的组合方案。排序式推荐在结构上无法解决，这是 GA 的用武之地。

形式化：
    max  Preference + Quality + Diversity + Novelty
    s.t. TotalDuration <= UserAvailableTime

染色体 = 选中的候选动漫下标子集（变长）。
交叉采用"并集 + 按性价比贪心裁剪回预算内"，变异为随机增删。
纯 Python 实现，与 services/ga.py 同风格，便于答辩讲解。
"""
import random
from database import get_conn

EP_MINUTES = 24          # 每集时长
MOVIE_MINUTES = 110      # 剧场版近似时长


def _to_set(s: str) -> set:
    return {x.strip() for x in (s or "").split(",") if x.strip()}


def load_candidates(keywords: str = "") -> list[dict]:
    """候选池：全目录；keywords（逗号分隔）命中 genre/theme 的作品加权优先保留。"""
    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM anime").fetchall()]
    conn.close()
    kws = [k.strip() for k in keywords.replace("，", ",").split(",") if k.strip()]
    out = []
    for a in rows:
        tags = set(a["genres"].split(",")) | set(a["themes"].split(","))
        match = sum(1 for k in kws if any(k in t for t in tags))
        minutes = MOVIE_MINUTES if a["an_type"] == "Movie" else a["episodes"] * EP_MINUTES
        out.append({**a, "minutes": minutes or MOVIE_MINUTES, "kw_match": match})
    if kws:  # 至少命中一个偏好关键词的才进入候选，避免计划跑题
        pool = [a for a in out if a["kw_match"] > 0]
        out = pool if len(pool) >= 3 else out
    return out


def _user_profile(user_id: int) -> dict[str, float]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.genres, a.themes FROM ratings r JOIN anime a ON a.id=r.anime_id
           WHERE r.user_id=? AND r.rating>=8
           UNION ALL
           SELECT a.genres, a.themes FROM favorites f JOIN anime a ON a.id=f.anime_id
           WHERE f.user_id=?""", (user_id, user_id)).fetchall()
    conn.close()
    profile: dict[str, float] = {}
    for r in rows:
        for t in (r["genres"] + "," + r["themes"]).split(","):
            t = t.strip()
            if t:
                profile[t] = profile.get(t, 0.0) + 1.0
    total = sum(profile.values()) or 1.0
    return {k: v / total for k, v in profile.items()}


def fitness(sel: list[int], cands: list[dict], profile: dict, budget: int) -> float:
    items = [cands[i] for i in sel]
    total_min = sum(c["minutes"] for c in items)
    if total_min > budget or not items:
        return -1e9  # 违反时长约束直接判负（交叉阶段也有修复，双保险）

    # Preference：与用户画像/关键词的匹配度
    pref = sum(profile.get(t, 0.0) for c in items for t in _to_set(c["genres"]) | _to_set(c["themes"]))
    pref = min(1.0, pref / max(1, len(items) * 1.5))

    # Quality：平均分归一化
    quality = sum(c["score"] for c in items) / (len(items) * 10.0)

    # Diversity：1 - 两两 genre Jaccard 均值（无配对时取中性 0.5，
    # 否则单部作品会白拿满分、GA 收敛到"一部超长番"）
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            gi, gj = _to_set(items[i]["genres"]), _to_set(items[j]["genres"])
            if gi | gj:
                pairs.append(len(gi & gj) / len(gi | gj))
    diversity = 1.0 - (sum(pairs) / len(pairs) if pairs else 0.5)

    # Novelty：越冷门越新颖
    novelty = 1.0 - sum(min(1.0, c["popularity"] / 4000000) for c in items) / len(items)

    # Utilization：预算利用率——人类期望计划尽量填满时间预算，而不是剩一半
    utilization = min(1.0, total_min / budget)

    return (0.35 * pref + 0.25 * quality + 0.15 * diversity
            + 0.1 * novelty + 0.15 * utilization)


def _repair(sel: list[int], cands: list[dict], budget: int) -> list[int]:
    """超预算时按（偏好分 × 评分 / 时长）性价比从低到高裁剪。"""
    total = sum(cands[i]["minutes"] for i in sel)
    while total > budget and sel:
        worst = min(sel, key=lambda i: (cands[i]["kw_match"] * 2 + cands[i]["score"])
                    / max(1, cands[i]["minutes"]))
        total -= cands[worst]["minutes"]
        sel.remove(worst)
    return sel


def make_plan(user_id: int, hours: float, keywords: str = "",
              generations: int = 80, pop_size: int = 40) -> dict:
    budget = int(hours * 60)
    cands = load_candidates(keywords)
    profile = _user_profile(user_id)
    n = len(cands)

    # 初始种群：随机子集（保证预算内）
    def rand_ind():
        k = random.randint(2, max(2, min(8, n)))
        sel = _repair(random.sample(range(n), k), cands, budget)
        return sel

    population = [rand_ind() for _ in range(pop_size)]
    best, best_fit = [], -1e9
    for _ in range(generations):
        scored = sorted(((fitness(s, cands, profile, budget), s) for s in population),
                        key=lambda t: t[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit, best = scored[0][0], scored[0][1][:]
        elite = [s for _, s in scored[: pop_size // 5]]
        nxt = elite[:]
        while len(nxt) < pop_size:
            a = max(random.sample(scored, min(6, len(scored))), key=lambda t: t[0])[1]
            b = max(random.sample(scored, min(6, len(scored))), key=lambda t: t[0])[1]
            child = list(dict.fromkeys(a + b))          # 并集交叉
            random.shuffle(child)
            child = _repair(child, cands, budget)        # 贪心裁剪回预算内
            if random.random() < 0.3 and n:              # 变异：随机增删
                pool = set(range(n)) - set(child)
                if pool and (len(child) < 2 or random.random() < 0.5):
                    child.append(random.choice(tuple(pool)))
                elif len(child) > 2:
                    child.remove(random.choice(child))
                child = _repair(child, cands, budget)
            nxt.append(child)
        population = nxt

    items = []
    for i in sorted(best, key=lambda i: -cands[i]["score"]):
        c = cands[i]
        items.append({
            "id": c["id"], "title": c["title"], "title_cn": c["title_cn"],
            "score": c["score"], "episodes": c["episodes"], "an_type": c["an_type"],
            "genres": [g for g in c["genres"].split(",") if g],
            "minutes": c["minutes"], "image_url": c.get("image_url", ""),
        })
    total_min = sum(c["minutes"] for c in items)
    return {
        "items": items,
        "total_minutes": total_min,
        "total_hours": round(total_min / 60, 1),
        "budget_hours": hours,
        "plan_fitness": round(best_fit, 4),
        "keywords": keywords,
    }
