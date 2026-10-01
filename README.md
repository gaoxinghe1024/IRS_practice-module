# AniLogic · Your ACG Friend

An anime discovery and viewing-plan system built for the NUS-ISS *Intelligent Reasoning Systems* practice module
(Group 10). AniLogic finds titles similar to one you like, recommends from your collections and your ratings,
answers natural-language requests with cited evidence, and plans what to watch within a time budget. It runs on
the Kaggle MyAnimeList 2023 snapshot (23,363 served titles, 24M ratings) or on a seeded synthetic world.

> **中文速览**：AniLogic 是 Group 10 的动漫发现与观看规划系统（FastAPI + SQLite + 原生 HTML/JS）。四个核心算法：
> ① 详情页相似推荐（加权属性 / 逻辑回归 / BGE 向量 / 融合）；② 收藏夹规则 + 偏好推荐（严格筛选，放宽需用户确认）；
> ③ 首页 ICF 召回 + GA 列表优化（含 MF / NCF / greedy 对照）；④ 聊天找番（意图解析 + 混合检索 + 带来源回答，LLM 可选）。
> 另有预算约束的观看计划、解释与反馈、冷启动。快速体验：`make setup && make seed && make serve`（模拟数据，
> 演示账号见 [docs/user-guide.md](docs/user-guide.md)）；真实数据流程见 [docs/operations.md](docs/operations.md)；
> 各功能状态见 [docs/delivery-status.md](docs/delivery-status.md)。

## What it does

| Surface | What you get | Method (default) |
|---|---|---|
| Title page | 5 similar titles, each with reasons | weighted five-attribute similarity; logistic, BGE-embedding and hybrid alternatives |
| Collection page | 8 recommendations matching the collection's taste and your strict filters | rule layer + profile scorer; GBDT / two-tower alternatives when collection data exists |
| Home | 10 personalised titles | item-based CF (exact adjusted cosine) recall → genetic-algorithm list optimisation; MF / NCF / greedy / MMR alternatives; cold-start modes |
| Chat | answers with title cards and cited sources; multi-turn constraints | deterministic intent parsing, FTS5 + dense hybrid retrieval (RRF), grounded answers; LLM optional and validated |
| Plan | titles, episode ranges, minutes, order and a budget check; save, edit, export | budgeted GA with prerequisites and progress; exact oracle for small pools |
| Status | what data and features are available now | capability statuses |

Relations, scene search and screenshot identification are built to data contracts and switch on when the data
exists; otherwise the status page says `limited_by_data`. See [docs/delivery-status.md](docs/delivery-status.md).

## Run from a clone (shipped state, no pipeline run)

The repository ships the raw Kaggle files, the active processed version of each mode, its trained artifacts and
both activated databases (about 1.8 GB, large files in Git LFS), so a clone serves exactly what the authors serve.

```bash
git lfs install                  # once per machine, before cloning (brew / apt install git-lfs)
git clone -b yyc-v1 <repo-url> && cd <repo>
uv sync --extra ml --extra llm   # same pinned packages as uv.lock
make serve-real                  # http://127.0.0.1:8000  (MAL 2023 data)
make serve                       # http://127.0.0.1:8000  (synthetic world, demo accounts: docs/user-guide.md)
```

Details: [docs/operations.md](docs/operations.md#cloning-and-deploying-the-shipped-state).

## Quick start (synthetic world, no downloads)

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

```bash
make setup      # dependencies (+ ml / llm / dev extras) and Playwright Chromium
make seed       # generate the synthetic world, train, index, activate, create demo accounts
make serve      # http://127.0.0.1:8000  (demo accounts: docs/user-guide.md)
```

`make setup-base` installs without the neural extras (LSA retrieval instead of BGE, no NCF).

## Real data

1. Download Kaggle `dbdmobile/myanimelist-dataset` (ODbL) into `data/raw/mal_2023/`.
2. Build and serve:

```bash
make validate-real preprocess-real split-real
make train build-index MODE=real
make activate-real
make serve-real
```

About 16 minutes end to end on an Apple M1 Pro. Step details, rollback, backups and optional Jikan enrichment:
[docs/operations.md](docs/operations.md). Switching modes needs no code changes.

## Docker

```bash
docker compose --profile setup run --rm setup   # first run: build the synthetic world into volumes
docker compose up --build                        # http://127.0.0.1:8000
```

(The compose file is provided but was not run on the delivery machine, which has no Docker.)

## Configuration

`configs/base.yaml` → `configs/<mode>.yaml` → environment variables `ANILOGIC_<SECTION>__<KEY>`. The optional LLM
is configured with `ANILOGIC_LLM__PROVIDER` / `__MODEL` / `__BASE_URL` and the secret `ANILOGIC_LLM_API_KEY`
(environment only). See [.env.example](.env.example).

## Tests and evaluation

```bash
make test          # unit + integration (builds a miniature synthetic world in a temp dir)
make e2e           # browser journeys 1-12 (Playwright)
make lint
make eval-smoke    # quick harness check (small samples; not a quality result)
make evaluate MODE=real && make export-evaluation MODE=real
```

Results: [docs/evaluation-results-real.md](docs/evaluation-results-real.md) (real data) and
[docs/evaluation-results-synthetic.md](docs/evaluation-results-synthetic.md) (implementation checks). Test report:
[docs/test-report.md](docs/test-report.md).

## Documentation

| Document | Content |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | using the app, demo accounts |
| [docs/delivery-status.md](docs/delivery-status.md) | what is implemented, tested and evaluated, and what is blocked |
| [docs/feature-traceability.md](docs/feature-traceability.md) | requirement → code → API → page → test → status |
| [docs/requirements-resolution.md](docs/requirements-resolution.md) | conflicts in the materials and the decisions taken |
| [docs/legacy-feature-audit.md](docs/legacy-feature-audit.md) | the replaced template, its features and security findings, migration |
| [docs/architecture.md](docs/architecture.md) | boundaries, data flow, serving state, security |
| [docs/algorithms.md](docs/algorithms.md) | formulas, hyperparameters, fallbacks |
| [docs/data-contracts.md](docs/data-contracts.md) | inputs, cleaning rules, supplementary and label files |
| [docs/api.md](docs/api.md), [docs/openapi.json](docs/openapi.json) | API reference |
| [docs/evaluation-protocol.md](docs/evaluation-protocol.md) | splits, metrics, leakage controls |
| [docs/operations.md](docs/operations.md) | setup, pipelines, activation, backup, troubleshooting |
| [DEVLOG.md](DEVLOG.md) | dated change log (team rule) |

## Repository layout

```
app/          FastAPI application: api/v1, services, recommendation, retrieval, chat, planning, graph, multimodal, web
pipelines/    command line and offline jobs (synthetic generator, real ingest, split, train, index, activate, Jikan, migration)
evaluation/   metrics, protocols, experiments, reports; runs/<run_id>/
configs/      layered configuration
migrations/   Alembic schema (incl. FTS5)
tests/        unit, integration, e2e
docs/         documentation, screenshots, evaluation results
```

## Data and licences

MyAnimeList 2023 snapshot: Kaggle `dbdmobile/myanimelist-dataset`, Open Database License (ODbL). Optional Jikan
data comes from the unofficial Jikan API (MyAnimeList data) and is cached offline; serving never calls it. Cover
images are loaded from MyAnimeList's CDN. The synthetic world is entirely fictional.
