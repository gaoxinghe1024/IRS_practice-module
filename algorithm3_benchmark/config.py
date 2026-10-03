# -*- coding: utf-8 -*-
"""
config.py — 全局配置：数据路径 + 实验协议参数 + GA 超参。

本文件是整个管线唯一的"参数中心"，所有脚本都 `import config as C` 读取这里，
所以想改协议（候选池档位、GA 权重、种子…）只需要改这一个文件。

协议来源：算法③幻灯片
  留一法四步；召回引擎 (ICF/SVD/NCF) × 重排层 (贪心/GA) 消融；
  GA 适应度 = 0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度。
"""
import os

# ------------------------------------------------------------------ 路径
HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(HERE, "data")
CF_DIR = os.path.join(DATA_ROOT, "cf")
GA_DIR = os.path.join(DATA_ROOT, "ga")
CANON_DIR = os.path.join(DATA_ROOT, "canonical")

OUT_DIR = os.path.join(HERE, "results")       # 结果数据 + 最终报告
CACHE_DIR = os.path.join(HERE, "cache")       # 引擎缓存（可随时删）
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------ 数据文件
F_TRAIN_NPZ = os.path.join(CF_DIR, "ratings_train.npz")
F_OBSERVED_NPZ = os.path.join(CF_DIR, "observed_train.npz")
F_USER_MAP = os.path.join(CF_DIR, "user_mapping.csv")
F_ANIME_MAP = os.path.join(CF_DIR, "anime_mapping.csv")
F_VALIDATION = os.path.join(CF_DIR, "ratings_validation.csv.gz")
F_ITEM_FEATURES = os.path.join(GA_DIR, "item_features.jsonl")
F_ANIME_RUNTIME = os.path.join(CANON_DIR, "anime_runtime.jsonl")

# ------------------------------------------------------------------ 实验协议
RANDOM_SEED = 42
COHORT_SIZE = 20000          # 参与评测的用户数（合格用户池的固定随机抽样）
MIN_TRAIN_RATINGS = 8        # 合格用户门槛：训练集至少 8 条评分
HIGH_RATING = 8.0            # 被藏起的那 1 条必须是 >= 8 分的高分记录
RATING_MIN = 1.0
RATING_MAX = 10.0

RECALL_POOL = 20             # 线上召回池 = Top-20（同时是 GA 的重排窗口深度）
TOP_K_OUT = 10               # 重排层最终输出 Top-10
HR_K = 5                     # HR@5 是主口径；报告同时给 HR@10

# ------------------------------------------------------------------ GA 超参
GA_POP = 60
GA_GENERATIONS = 30
GA_TOURNAMENT = 3
GA_CROSSOVER = 0.8
GA_MUTATION = 0.30
GA_ELITE = 4
# 幻灯片口径：Fitness = 0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度
GA_W_RELEVANCE = 0.4
GA_W_DIVERSITY = 0.3
GA_W_NOVELTY = 0.2
GA_W_FRESH = 0.1
GA_FITNESS_MODE = "slide"    # "slide" | "handoff"

# ------------------------------------------------------------------ 引擎超参
SVD_DIM = 32                 # 幻灯片：用户与作品共享 32 维空间
SVD_EPOCHS = 12
SVD_LR = 0.01
SVD_L2 = 0.05
SVD_BATCH = 16384

NCF_EMB = 32
NCF_EPOCHS = 8
NCF_LR = 0.002
NCF_BATCH = 16384
NCF_NEG = 4                  # 每个正例配 4 个负例
NCF_POS_THRESHOLD = 8.0      # 隐式正例 = 评分 >= 8
NCF_MAX_POS = 1500000        # 每轮正例上限（CPU/GPU 预算）
NCF_PATIENCE = 2

ICF_NEIGHBOURS = 100         # 每个作品保留的 Top-K 邻居
ICF_BLOCK = 256              # 相似度分块（内存上限）
ICF_MAX_SEED_ITEMS = 500     # 构造查询向量时的画像上限

DEVICE = "cuda"
