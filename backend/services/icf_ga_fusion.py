# -*- coding: utf-8 -*-
"""
算法③（核心推荐）：ICF + GA Fusion
====================================
对应界面：主页「为你推荐」（核心推荐位）。

融合链路（本文件只做编排，不重复实现算法本体）：
    算法② Item-based Collaborative Filtering（services/cf.py）
        —— 基于用户【真实评分】计算 item-item 相似度，召回 Top-20 候选
    算法③ Genetic Algorithm（services/ga.py）
        —— Fitness = 0.4·Relevance + 0.3·Personalization + 0.2·Diversity + 0.1·Novelty
           对候选做列表级重排，输出最优 Top-K 与贪心基线对比
"""
from services import cf, ga


def home_recommend(user_id: int, top_k: int = 5) -> dict:
    """主页核心推荐：算法② ICF 召回 → 算法③ GA 重排。"""
    candidates = cf.cf_candidates(user_id, top_n=20)          # 算法②：ICF 召回
    optimized, best_fit, baseline_fit = ga.optimize(          # 算法③：GA 重排
        candidates, user_id, top_k=top_k)
    return {
        "items": optimized,
        "algorithm": "ICF + GA Fusion（算法②协同过滤召回 + 算法③遗传算法重排）",
        "ga": {"best_fitness": round(best_fit, 4), "baseline_fitness": round(baseline_fit, 4),
               "improved": round(best_fit - baseline_fit, 4),
               "weights": ga.W},
    }
