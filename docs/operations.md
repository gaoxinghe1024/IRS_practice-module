# Operations

Running, rebuilding and troubleshooting AniLogic. The timings were measured on an Apple M1 Pro (8 cores, 16 GB
RAM) with the full real dataset and no user sampling.

## 1. Environment

| Need | Version / note |
|---|---|
| Python | 3.11-3.13 (developed and tested on 3.12) |
| uv | dependency manager; `uv.lock` pins every package. `requirements.txt` is generated from it (`make requirements`); pip users run `pip install -r requirements.txt && pip install --no-deps -e .` |
| Extras | `ml` (torch, sentence-transformers: BGE retrieval, NCF, two-tower), `llm` (anthropic SDK), `dev` (pytest, ruff, playwright) |
| Disk | raw Kaggle files 1.2 GB; per real version about 220 MB processed + 235 MB artifacts; business DB about 120 MB; Hugging Face cache about 1.6 GB (BGE model) |
| Browser tests | Chromium via Playwright (`make setup` installs it) |

```bash
make setup          # uv sync --extra ml --extra llm --extra dev + Playwright Chromium
make setup-base     # without the neural extras: LSA retrieval, no NCF / BGE / two-tower
```

**macOS / iCloud Drive note.** If the project lives in an iCloud-synced folder, macOS can set the `UF_HIDDEN` flag
on the editable-install `.pth` file inside `.venv`, and Python then ignores it (`ModuleNotFoundError: app`).
`make setup` clears the flag (`chflags nohidden .venv/lib/python*/site-packages/*.pth`), and every Makefile target
uses `python -m ...` from the project root, which does not depend on the `.pth` file.

## 2. Configuration

Layers: `configs/base.yaml` → `configs/<mode>.yaml` → environment variables `ANILOGIC_<SECTION>__<KEY>` (double
underscore for nesting), for example `ANILOGIC_RECOMMENDATION__HOME__RERANKER=greedy`. The mode comes from
`ANILOGIC_DATASET_MODE` (`synthetic` | `real`). Secrets are read from the environment only
(`ANILOGIC_LLM_API_KEY`); see `.env.example`. Never commit `.env`.

## 3. Synthetic mode (no downloads needed)

```bash
make seed           # generate → preprocess → split → train → build-index → activate → demo accounts
make serve          # http://127.0.0.1:8000
```

Demo accounts (synthetic database only) are listed in `docs/user-guide.md`. `make clean-synthetic` removes
the generated world, its artifacts and its database.

## 4. Real mode

1. Put the Kaggle files into `data/raw/mal_2023/` (see `data/README.md`).
2. Run the pipeline. Each step is recorded under `var/real/pipeline_runs/`. Exit codes: 0 ok, 2 invalid input,
   3 precondition missing, 1 unexpected failure.

| Step | Command | Measured |
|---|---|---|
| validate | `make validate-real` | reads and hashes the 1.2 GB of raw files (`build-all` runs a fast header check, about 1 s) |
| preprocess | `make preprocess-real` | 13 s, peak about 0.5 GB RSS (24.3M ratings in 100k-row chunks) |
| split | `make split-real` | 85 s |
| train | `make train MODE=real` | about 12 min: ICF 82 s, MF 184 s, NCF 434 s (12 epochs, Apple MPS), BGE attribute embeddings 25 s |
| build index | `make build-index MODE=real` | about 3 min (42.5k documents encoded with BGE on MPS) |
| activate | `make activate-real` | about 15 s (catalogue + documents + FTS5 in one transaction) |
| serve | `make serve-real` | loads the serving state in a few seconds |

Or run everything at once: `python -m pipelines.cli build-all --mode real`.

**Laptop sampling.** `ANILOGIC_PIPELINE__SAMPLE_USERS_FRACTION=0.25` keeps a seeded quarter of the offline users.
The fraction is part of the dataset version and is recorded in the preprocessing report. The delivered
evaluation used all users.

**Models.** The BGE model (`BAAI/bge-small-en-v1.5`) is downloaded by sentence-transformers on the first
`build-index` / `train` and cached under `~/.cache/huggingface` (or `HF_HOME`). After that, `HF_HUB_OFFLINE=1`
runs fully offline. Without the `ml` extra the index falls back to LSA (TF-IDF + SVD), which is reported under
its own name.

## 5. Evaluation

```bash
make evaluate MODE=real            # all experiments; each run gets its own directory under evaluation/runs/
make export-evaluation MODE=real   # docs/evaluation-results-real.md from the latest finished runs
make eval-smoke                    # quick harness check (40 users, 1 seed; tagged smoke, never exported)
```

The ICF parameters are tuned on validation data by the homepage experiment. After retraining, run
`python -m pipelines.cli evaluate --mode real --experiment homepage`, then `make apply-tuning MODE=real`. That
writes the chosen damping and user-mean prior weight into `configs/real.yaml` with the run id, and after it you
run the full evaluation, so that every experiment sees the served configuration. On the M1 Pro the full real
evaluation takes about an hour, most of it in the homepage and GA-ablation experiments.

## 6. Versions, activation and rollback

* `data/processed/real/latest.json` names the newest processed version; `artifacts/real/active.json` names the
  served one.
* Activation never deletes user data. Titles missing from a new version stay referenced but hidden.
* **Rollback:** `python -m pipelines.cli activate --mode real --version <older-version>` (its processed data and
  artifacts must still exist). A running server notices the change and swaps its serving state.
* The serving state checks every artifact against the activation record. If a file was modified, the status page
  shows `artifact_mismatch`, and the fix is to re-run `build-index` / `train` and activate again.
* Old versions are not deleted automatically. When you no longer need one for rollback or for reproducing an
  evaluation run, remove its `data/processed/<mode>/<version>` and `artifacts/<mode>/<version>` directories.

## 7. Backup and restore

The business database is `var/<mode>/anilogic.sqlite3` (WAL mode). Back it up consistently while the app runs:

```bash
sqlite3 var/real/anilogic.sqlite3 ".backup 'backups/anilogic-$(date +%Y%m%d).sqlite3'"
```

To restore, stop the app, copy the backup to `var/real/anilogic.sqlite3` (remove any `-wal` / `-shm` files next
to it) and start again. Processed data and artifacts can be rebuilt from the raw files; user data cannot, so
back up `var/`.

## 8. Administration

* Make an account an administrator (there is no default admin):
  `python -m pipelines.cli make-admin --mode real <username>`. Admins get `/dev` (algorithm comparisons,
  run traces) and `POST /api/v1/admin/reload`.
* Capability status: `/status` page, `GET /api/v1/system/status`, or `python -m pipelines.cli status --mode real`.
* Logs: JSON lines on stdout with a request id; chat text is not logged unless `app.log_chat_text: true`.

## 9. Optional services

### Jikan enrichment (real mode)

```bash
make enrich-jikan          # top 400 popular titles; resumable; responses cached in data/enriched/jikan/
```

Requests are spaced 1.1 s apart (below Jikan's documented limits), 429 responses honour `Retry-After`, and 5xx
responses are collected by the next pass instead of being retried in a tight loop. Serving never calls Jikan.
After enrichment, re-run preprocess → activate: the cache is part of the dataset version. **Status during
this project:** MyAnimeList behind Jikan returned HTTP 504 for most requests (2026-09-30), so only 33 titles got
`/full` data. Re-running the command later continues where it stopped.

### LLM for chat

Set `ANILOGIC_LLM__PROVIDER` (`openai_compatible` with `BASE_URL` + `MODEL`, or `anthropic` with `MODEL`) and
`ANILOGIC_LLM_API_KEY`, then restart. Without them chat runs in local mode, and every feature still works. The
evaluation's Plain LLM vs RAG comparison is `blocked_by_credentials` until a key is configured. Running it calls
a paid API, so it is started only with explicit approval.

### Legacy account migration

`make migrate-legacy MODE=real` runs a dry run against `backend/anime.db` and writes
`var/real/legacy_migration_dryrun.json`. `ARGS=--apply` writes, after copying the target database to a
timestamped backup. See `docs/legacy-feature-audit.md` §4.

## 10. Docker

```bash
docker compose --profile setup run --rm setup     # build the synthetic world into the volumes (first time)
docker compose up --build                          # app on http://127.0.0.1:8000
```

The image contains code only. Data, artifacts, the business database and the model cache live in named volumes,
and `data/raw` is mounted read-only for real mode. The container runs as a non-root user with a health check on
`/api/v1/system/health`. *Not verified on the delivery machine: Docker is not installed there. The non-Docker
path above is the tested one.*

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: app` | hidden `.pth` (see §1); run from the project root with `python -m` |
| Status page `catalog: missing_data` | nothing activated for this mode: run the pipeline (`make seed` or the real steps) |
| `artifact_mismatch` | artifacts changed after activation: rebuild and activate again |
| `retrieval_dense: degraded` / LSA shown | the `ml` extra is missing or the model is not cached; `make setup`, then `build-index` + `activate` |
| Chat says "Local mode" | no LLM configured (expected); see §9 |
| `database is locked` in logs | heavy concurrent writes; writes retry with a bounded back-off. If it persists, check for a long-running external `sqlite3` session |
| `enrich-jikan` shows many 504s | upstream MyAnimeList outage; run it again later |
| Browser tests fail to start | `python -m playwright install chromium` |
