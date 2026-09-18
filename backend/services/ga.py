# -*- coding: utf-8 -*-
"""
算法③（组件）：Evolutionary Computing → Genetic Algorithm（遗传算法）
======================================================================
对应界面：主页「为你推荐」的重排层（由 services/icf_ga_fusion.py 与 ICF 融合编排）；
另有一个附加应用：观看计划优化（services/watchplan.py，GA 组合优化）。

Fitness = w1*Relevance + w2*Personalization + w3*Diversity + w4*Novelty
默认权重 0.4 / 0.3 / 0.2 / 0.1（与项目文档一致）。

染色体 = 一个 Top-K 推荐方案的动漫 id 序列。
纯 Python 实现（random + set），无重型依赖，答辩可逐行讲解。
"""
import random
import math
from database import get_conn

W = dict(relevance=0.4, personalization=0.3, diversity=0.2, novelty=0.1)


def _to_set(s: str) -> set:
    return {x.strip() for x in (s or "").split(",") if x.strip()}


def _user_profile(user_id: int) -> dict:
    """Personalization 用：用户评分>=8 的作品的 genre 分布画像。"""
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.genres FROM ratings r JOIN anime a ON a.id=r.anime_id
           WHERE r.user_id=? AND r.rating>=8""",
        (user_id,)).fetchall()
    fav = conn.execute(
        """SELECT a.genres FROM favorites f JOIN anime a ON a.id=f.anime_id WHERE f.user_id=?""",
        (user_id,)).fetchall()
    conn.close()
    profile: dict[str, float] = {}
    for r in rows + fav:
        for g in _to_set(r["genres"]):
            profile[g] = profile.get(g, 0.0) + 1.0
    total = sum(profile.values()) or 1.0
    return {k: v / total for k, v in profile.items()}


def fitness(chromosome: list[int], candidates: list[dict], profile: dict[str, float],
            max_pop: float) -> float:
    cand_by_id = {c["id"]: c for c in candidates}
    items = [cand_by_id[i] for i in chromosome]

    # R：候选本身的相关性得分（ICF 分数归一化）
    max_cf = max((c.get("cf_score", 0) for c in candidates), default=1.0) or 1.0
    relevance = sum(c.get("cf_score", 0) for c in items) / (len(items) * max_cf)

    # P：列表与用户画像的 genre 匹配度
    if profile:
        hits = sum(profile.get(g, 0.0) for c in items for g in _to_set(c["genres"]))
        p_raw = hits / max(1, sum(len(_to_set(c["genres"])) for c in items))
        personalization = min(1.0, p_raw * 2.0)
    else:
        personalization = 0.5

    # D：多样性 = 1 - 两两 genre Jaccard 均值（向量分不开的同类番会拉低 D）
    pairwise = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            gi, gj = _to_set(items[i]["genres"]), _to_set(items[j]["genres"])
            if gi | gj:
                pairwise.append(len(gi & gj) / len(gi | gj))
    diversity = 1.0 - (sum(pairwise) / len(pairwise) if pairwise else 0.0)

    # N：新颖性 = 1 - 归一化热度均值（越冷门越新颖）
    novelty = 1.0 - sum(min(1.0, c.get("popularity", 0) / max_pop) for c in items) / max(1, len(items))

    return (W["relevance"] * relevance + W["personalization"] * personalization
            + W["diversity"] * diversity + W["novelty"] * novelty)


def optimize(candidates: list[dict], user_id: int, top_k: int = 5,
             generations: int = 60, pop_size: int = 30) -> tuple[list[dict], float, float]:
    """GA 重排：返回 (优化后的 Top-K 列表, 最优 fitness, 基线 fitness[按相关性贪心])。"""
    if len(candidates) <= top_k:
        baseline = [dict(c, fitness=0.0) for c in candidates]
        return baseline, 0.0, 0.0

    conn = get_conn()
    max_pop = conn.execute("SELECT MAX(popularity) FROM anime").fetchone()[0] or 1
    conn.close()
    profile = _user_profile(user_id)
    cand_ids = [c["id"] for c in candidates]

    # 基线：纯相关性排序（用于演示 GA 带来的多样性提升）
    greedy_ids = [c["id"] for c in sorted(candidates, key=lambda c: c.get("cf_score", 0), reverse=True)[:top_k]]
    baseline_fit = fitness(greedy_ids, candidates, profile, max_pop)

    # 初始种群：随机抽样 + 贪心解保底
    population = [greedy_ids[:]]
    for _ in range(pop_size - 1):
        population.append(random.sample(cand_ids, top_k))

    best = greedy_ids[:]
    best_fit = baseline_fit
    for _ in range(generations):
        scored = [(fitness(ch, candidates, profile, max_pop), ch) for ch in population]
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit, best = scored[0][0], scored[0][1][:]
        # 精英保留 + 锦标赛选择
        elite = [ch for _, ch in scored[: pop_size // 5]]
        nxt = elite[:]
        while len(nxt) < pop_size:
            a = max(random.sample(scored, min(5, len(scored))), key=lambda t: t[0])[1]
            b = max(random.sample(scored, min(5, len(scored))), key=lambda t: t[0])[1]
            # 交叉：两点交叉 + 去重
            i, j = sorted(random.sample(range(top_k), 2))
            child = a[:i] + b[i:j] + a[j:]
            child = list(dict.fromkeys(child))
            pool = set(cand_ids) - set(child)
            while len(child) < top_k and pool:
                pick = random.choice(tuple(pool))
                child.append(pick)
                pool.discard(pick)
            # 变异
            if random.random() < 0.15 and cand_ids:
                k = random.randrange(top_k)
                child[k] = random.choice(cand_ids)
                child = list(dict.fromkeys(child))
                pool = set(cand_ids) - set(child)
                while len(child) < top_k and pool:
                    pick = random.choice(tuple(pool))
                    child.append(pick)
                    pool.discard(pick)
            nxt.append(child)
        population = nxt

    cand_by_id = {c["id"]: c for c in candidates}
    result = [dict(cand_by_id[i], ga_fitness=round(best_fit, 4)) for i in best]
    return result, best_fit, baseline_fit
