# AniLogic — System Overview

AniLogic is an anime discovery and viewing-planning web application built for the
NUS-ISS IRS course (Group 18). It combines four recommendation algorithms, a
grounded RAG chat assistant, and a budget-constrained viewing planner on top of a
single FastAPI service with a server-rendered frontend.

This document maps every part of the `system code` folder: where things live and
what they do. For hardware/software requirements see `SYSTEM_REQUIREMENTS.md`;
for how to run the system see `STARTUP_GUIDE.md`.

---

## 1. Repository layout

```
system code/
├── app/                  Application source code (FastAPI service, everything runtime)
├── configs/              YAML configuration, layered: base -> <mode> -> env vars
├── pipelines/            Offline data pipelines (ingest, train, index, activate, seed)
├── evaluation/           Evaluation harness: experiments, metrics, run records
├── migrations/           Alembic database schema migrations
├── tests/                Unit, integration and e2e test suites
├── data/                 Raw / processed / enriched datasets (large, mostly Git LFS)
├── artifacts/            Trained models, indexes and the frozen algorithm-2 bundle
├── var/                  Generated runtime data: SQLite database per dataset mode
├── pyproject.toml        Package metadata, dependencies, optional extras
├── requirements.txt      Pinned requirements (regenerated from uv.lock)
├── uv.lock               Exact dependency lockfile used by uv / the Docker image
├── alembic.ini           Migration tool configuration
├── Makefile              All operational commands (setup, serve, test, eval, ...)
├── Dockerfile            Application image (contains code only; data is mounted)
├── compose.yaml          Docker Compose wrapper around the image
├── .env.example          Template of all supported environment variables
├── .gitignore / .gitattributes   Repo hygiene + Git LFS tracking rules
└── SYSTEM_REQUIREMENTS.md, STARTUP_GUIDE.md, SYSTEM_OVERVIEW.md (this file)
```

---

## 2. `app/` — the running application

| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI entrypoint: lifespan hook (migrations + catalog load), middleware, router registration |
| `app/core/config.py` | Settings loader (`configs/base.yaml` -> `configs/<mode>.yaml` -> `ANILOGIC_*` env vars); also resolves project paths |
| `app/core/security.py` | Password hashing (argon2), sessions, CSRF tokens |
| `app/core/errors.py` / `logging.py` / `ids.py` / `text.py` | Error envelope, structured logging, ID helpers, text normalization |
| `app/db/models.py` | SQLAlchemy ORM models for all business tables (users, collections, plans, feedback, chat sessions) |
| `app/db/session.py` | Database engine/session factory (SQLite via `var/<mode>/anilogic.sqlite3`) |
| `app/db/vectors.py` | Vector storage helpers used by dense retrieval |

### `app/api/v1/` — REST API (mounted under `/api/v1`)

| Module | Endpoints |
|---|---|
| `auth.py` | Register / login / logout |
| `me.py` | Current user profile and preferences |
| `anime.py` | Anime detail, search, **algorithm 1** similar-anime recommendations |
| `recommendations.py` | **Algorithm 3** homepage recommendations |
| `collections.py` | User collections CRUD + **algorithm 2** collection recommendations |
| `chat.py` | Chat sessions and messages (**algorithm 4**, grounded RAG answers) |
| `plans.py` | Viewing plan preview / save / list / export (markdown) |
| `scenes.py` | Scene lookup for titles that have scene data |
| `images.py` | Image proxy for external cover art |
| `admin.py` | Dev/admin utilities |
| `system.py` | `/health` and `/status` (capability matrix of the running instance) |

### Recommendation stack

| Path | Purpose |
|---|---|
| `app/recommendation/similarity/algorithm1.py` | Algorithm 1: knowledge-based pair scoring (tags/themes/studios/relations) |
| `app/recommendation/similarity/logistic.py`, `weighted.py`, `embedding.py` | Alternative similarity scorers (legacy logistic, weighted-blend, embedding) |
| `app/recommendation/collection/` | Algorithm 2 training-side implementations: gradient boosting (LightGBM-format trees parsed in pure numpy), manual formulas (A0/B0/B1), two-tower |
| `app/recommendation/algorithm2/` | Algorithm 2 serving side: `bundle.py` loads the frozen model package with sha256 verification, `service.py` resolves `deploy_seed`, `scorers.py`, `gbdt_tree.py`, `two_tower.py` |
| `app/recommendation/recall/` | Candidate generation: ICF (item collaborative filtering), matrix factorization, NCF |
| `app/recommendation/rerank/` | Genetic-algorithm reranker (`ga.py`) and its multi-objective `objective.py` |
| `app/recommendation/constraints.py`, `profiles.py`, `features.py`, `explanations.py` | Hard filters, taste profiles, shared feature builders, human-readable explanations |

### Other runtime modules

| Path | Purpose |
|---|---|
| `app/chat/intent.py` | Constraint extraction from natural-language queries (episode count, duration, studio, ...) |
| `app/chat/grounded_answer.py` | Evidence-grounded answer composition (local template mode) |
| `app/chat/conversation.py` | Multi-turn context handling |
| `app/chat/llm.py` | Optional LLM adapter (OpenAI-compatible or Anthropic); absent a key, chat stays in local template mode |
| `app/services/serving.py` | Startup catalog snapshot: reads the whole catalog from SQLite into memory, verifies artifact manifests, exposes the serving state used by all recommenders |
| `app/services/` (others) | `auth`, `catalog`, `chat`, `plans`, `recommend`, `user_data` — business logic between API and models |
| `app/retrieval/` | Evidence retrieval for chat: lexical (FTS5), dense (BGE when the ml extra is installed), fusion, document store |
| `app/planning/` | Viewing-plan engine: duration estimates, feasibility, optimizer, schedule building, markdown export |
| `app/graph/relations.py` | Franchise/relation graph lookups |
| `app/multimodal/image.py` | Cover-image feature fallbacks |
| `app/web/pages.py` | Server-rendered page routes (`/`, `/discover`, `/collections`, `/chat`, `/plan`, ...) |
| `app/web/templates/` | Jinja2 templates (one per page + `base.html`) |
| `app/web/static/` | Frontend assets: `css/style.css`, `js/common.js`, `js/pages/*`, images |
| `app/web/i18n/en.json` | UI strings (English; the UI language pack) |

---

## 3. `configs/` — configuration

| File | Purpose |
|---|---|
| `base.yaml` | Defaults for every section (app host/port, paths, recommendation, chat, planner) |
| `synthetic.yaml` | Overrides for the fictional demo world |
| `real.yaml` | Overrides for the Kaggle MAL-2023 dataset mode; points `recommendation.collection.algorithm2_run` at the frozen bundle in `artifacts/real/algorithm2-run/real-main-01` |

Environment variables (`ANILOGIC_<SECTION>__<KEY>`) override YAML values; see `.env.example`.

---

## 4. `pipelines/` — offline data pipeline

| File | Purpose |
|---|---|
| `cli.py` | Single CLI entrypoint dispatching all pipeline stages (`python -m pipelines --help`) |
| `real_ingest.py` | Ingest Kaggle MAL-2023 CSVs from `data/raw/mal_2023/` |
| `package_import.py` | Import the team's curated data package (`../Data`, optional) |
| `cleaning.py`, `processed_io.py` | Cleaning and intermediate parquet I/O |
| `split.py` | Train/validation/test splits (records the split report) |
| `fit.py`, `build_index.py` | Train recommenders and build retrieval indexes into `artifacts/` |
| `activate.py` | Activate a dataset version: writes the manifest consumed at startup |
| `validate_data.py` | Data-contract validation for optional supplementary/evaluation files |
| `synthetic.py`, `demo_seed.py` | Generate the fictional demo world and demo accounts |
| `jikan.py` | Optional offline Jikan API enrichment (real mode, rate-limited) |
| `legacy_migration.py` | Legacy account migration tool |

Typical order (real mode): ingest -> preprocess -> split -> train -> build-index -> activate.

---

## 5. `evaluation/` — experiment harness

| Path | Purpose |
|---|---|
| `runner.py`, `experiments/` | Dispatch and implement each experiment (similar, homepage, collection, explicit, ga_ablation, planner, chat, system, multimodal) |
| `common.py` | Run folders under `evaluation/runs/<timestamp>-<mode>-<name>-<hash>/`; writes metrics plus a human-readable `report.md` |
| `metrics.py`, `protocols.py`, `sandbox.py` | Metric definitions, evaluation protocols, sandboxed execution |
| `reports.py` | Collects the latest run per experiment into summary reports; also reads the algorithm-2 decision record |
| `runs/` | Shipped experiment records (metrics + reports). These are generated evidence, not hand-written docs — `reports.py` reads them |

---

## 6. Data, artifacts, var — what the system actually reads at runtime

| Path | Contents |
|---|---|
| `data/raw/mal_2023/` | Kaggle "Anime Dataset 2023" CSVs (source of truth for real mode) |
| `data/processed/`, `data/enriched/`, `data/synthetic/` | Pipeline intermediates and optional enrichment |
| `data/evaluation/` | Human-evaluation inputs |
| `var/real/anilogic.sqlite3` | The business SQLite database (real mode). Loaded fully into memory at startup; edits show after restart |
| `artifacts/real/real-034ebd6a74/` | Trained models and indexes (doc vectors, NCF, MF, item attributes, catalog parquet). Startup verifies each file's sha256 against `manifest.json` |
| `artifacts/real/algorithm2-run/real-main-01/` | Frozen algorithm-2 serving bundle (5 files, ~3.9 MB): `freeze_manifest.json`, `decision.json`, `models/B0_seed42.txt`, `models/B1_seed42.txt`, `models/C0_seed42.pt`. Only seed-42 models are loaded; their sha256 is checked against `freeze_manifest.json`. If the algorithm-2 experiment is ever re-run, this folder must be refreshed from the new run |
| `artifacts/synthetic/...` | Same idea for the synthetic world |

---

## 7. Tests

| Path | Purpose |
|---|---|
| `tests/unit/` | Fast unit tests (scorers, planner, config, chat intent, ...) |
| `tests/integration/` | Full-stack tests against miniature generated datasets |
| `tests/e2e/` | Browser end-to-end tests (Playwright; needs the `dev` extra) |
| `tests/helpers.py`, `tests/conftest.py` | Shared fixtures and dataset builders |

Run with `make test` (see `STARTUP_GUIDE.md`).
