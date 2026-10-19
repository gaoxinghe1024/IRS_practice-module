# -*- coding: utf-8 -*-
"""
config.py — Global configuration: data paths, experiment protocol parameters, GA hyper-parameters.

Single parameter center for the whole pipeline; every script reads it via
`import config as C`, so protocol changes (pool grid, GA weights, seeds, ...)
only need to happen here.
"""
import os

# ------------------------------------------------------------------ Paths
HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(HERE, "data")
CF_DIR = os.path.join(DATA_ROOT, "cf")
GA_DIR = os.path.join(DATA_ROOT, "ga")
CANON_DIR = os.path.join(DATA_ROOT, "canonical")

OUT_DIR = os.path.join(HERE, "results")       # result data + final reports
CACHE_DIR = os.path.join(HERE, "cache")       # engine cache (safe to delete)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------ Data files
F_TRAIN_NPZ = os.path.join(CF_DIR, "ratings_train.npz")
F_OBSERVED_NPZ = os.path.join(CF_DIR, "observed_train.npz")
F_USER_MAP = os.path.join(CF_DIR, "user_mapping.csv")
F_ANIME_MAP = os.path.join(CF_DIR, "anime_mapping.csv")
F_VALIDATION = os.path.join(CF_DIR, "ratings_validation.csv.gz")
F_ITEM_FEATURES = os.path.join(GA_DIR, "item_features.jsonl")
F_ANIME_RUNTIME = os.path.join(CANON_DIR, "anime_runtime.jsonl")

# ------------------------------------------------------------------ Experiment protocol
RANDOM_SEED = 42
COHORT_SIZE = 20000          # number of users in the evaluation cohort (fixed random sample of eligible users)
MIN_TRAIN_RATINGS = 8        # eligibility threshold: at least 8 ratings in the training set
HIGH_RATING = 8.0            # the hidden item must be a high rating (>= 8.0)
RATING_MIN = 1.0
RATING_MAX = 10.0

RECALL_POOL = 20             # online recall pool = Top-20 (also the rerank window depth for GA)
TOP_K_OUT = 10               # rerank layer outputs the final Top-10
HR_K = 5                     # HR@5 is the primary metric; HR@10 is also reported

# ------------------------------------------------------------------ GA hyper-parameters
GA_POP = 60
GA_GENERATIONS = 30
GA_TOURNAMENT = 3
GA_CROSSOVER = 0.8
GA_MUTATION = 0.30
GA_ELITE = 4
# Slide convention: Fitness = 0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 freshness
GA_W_RELEVANCE = 0.4
GA_W_DIVERSITY = 0.3
GA_W_NOVELTY = 0.2
GA_W_FRESH = 0.1
GA_FITNESS_MODE = "slide"    # "slide" | "handoff"

# ------------------------------------------------------------------ Engine hyper-parameters
SVD_DIM = 32                 # slide: users and items share a 32-dim latent space
SVD_EPOCHS = 12
SVD_LR = 0.01
SVD_L2 = 0.05
SVD_BATCH = 16384

NCF_EMB = 32
NCF_EPOCHS = 8
NCF_LR = 0.002
NCF_BATCH = 16384
NCF_NEG = 4                  # 4 negatives per positive sample
NCF_POS_THRESHOLD = 8.0      # implicit positive = rating >= 8
NCF_MAX_POS = 1500000        # per-epoch positive cap (CPU/GPU budget)
NCF_PATIENCE = 2

ICF_NEIGHBOURS = 100         # Top-K neighbours kept per item
ICF_BLOCK = 256              # similarity blocking (memory cap)
ICF_MAX_SEED_ITEMS = 500     # profile cap when building the query vector

DEVICE = "cuda"
