# AniLogic — Startup Guide

How to install, configure, start and verify the system. Requirements are listed
in `SYSTEM_REQUIREMENTS.md`; a map of every file is in `SYSTEM_OVERVIEW.md`.

All commands below are run from the `system code` directory.

---

## 1. Install dependencies

### Option A — uv (recommended, matches `uv.lock` exactly)

```bash
uv sync --extra ml --extra llm --extra dev   # full environment
# or the minimal core only:
uv sync
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e .
# optional extras:
pip install torch sentence-transformers lightgbm   # ml extra
pip install anthropic                              # llm extra
```

Or simply `make setup` (full) / `make setup-base` (core only).

> The system runs with core dependencies only: algorithm 2's shipped LightGBM
> trees are parsed in pure numpy, and chat falls back to local template answers
> without an LLM key. Dense retrieval degrades to lexical FTS5 without the
> `ml` extra.

---

## 2. Choose a dataset mode

| Mode | What it is | What must exist |
|---|---|---|
| `real` | Kaggle MAL-2023 snapshot (24,905 anime) | `var/real/anilogic.sqlite3` + `artifacts/real/real-034ebd6a74/` (both shipped) |
| `synthetic` | Generated fictional demo world | `make seed` builds it if absent |

Select with the environment variable `ANILOGIC_DATASET_MODE=real` (or `synthetic`),
or `MODE=real` with the Makefile targets.

---

## 3. Start the server

### Makefile

```bash
make serve MODE=real            # http://127.0.0.1:8000
make serve MODE=real HOST=0.0.0.0 PORT=8899   # custom bind
```

### Plain Python

```bash
ANILOGIC_DATASET_MODE=real python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows (cmd/PowerShell):

```powershell
$env:ANILOGIC_DATASET_MODE = "real"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Docker

```bash
# image contains code only; data/artifacts/var are mounted from this folder
ANILOGIC_DATASET_MODE=real docker compose up --build
```

---

## 4. What startup does

1. Runs Alembic migrations on the mode's database (`var/<mode>/anilogic.sqlite3`).
2. Loads the entire catalog from SQLite into memory (a snapshot — edits to the
   database file while running are not visible until restart).
3. Verifies every artifact file's sha256 against `artifacts/real/real-*/manifest.json`.
4. Loads the frozen algorithm-2 bundle from
   `artifacts/real/algorithm2-run/real-main-01/` and checks the seed-42 models'
   hashes against its `freeze_manifest.json`.

First startup takes roughly 10-20 seconds in real mode. The log line
`serving state loaded` means the catalog snapshot is ready.

---

## 5. Verify it is running

```bash
curl http://127.0.0.1:8000/api/v1/system/health
# -> {"ok":true,...}

curl http://127.0.0.1:8000/api/v1/system/status
# capability matrix: catalog, each algorithm, retrieval, llm, ...
```

Then open `http://127.0.0.1:8000/` in a browser:

1. Click **Register** and create an account (there are no pre-seeded users in
   real mode; synthetic mode has demo accounts from `make seed`).
2. Browse `/discover`, open an anime, add titles to a collection, ask the chat
   for recommendations, and build a viewing plan under `/plan`.
3. `/status` shows the same capability matrix in the UI.
4. `/dev` is the admin page — only visible to accounts with `is_admin`.

---

## 6. Common operations (Makefile)

| Command | Purpose |
|---|---|
| `make test` | Unit + integration tests (builds a miniature synthetic world in a temp dir) |
| `make test-unit` / `make test-integration` | Subsets |
| `make eval-smoke` | Quick sanity run of every evaluation experiment |
| `make evaluate MODE=real` | Full evaluation suite |
| `make status` | Current configuration summary |
| `make build-resources MODE=<mode>` | Retrain models / rebuild indexes, then activate |
| `make requirements` | Regenerate pinned `requirements.txt` from `uv.lock` |
| `make clean-synthetic` | Remove generated synthetic data/artifacts/database |

Pipeline stages (only needed when rebuilding data from raw CSVs) are dispatched
through `python -m pipelines --help` (ingest -> preprocess -> split -> train ->
build-index -> activate).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `collection_scorer: degraded` or `A0` in `/status` | Frozen algorithm-2 bundle missing/hash mismatch — check `artifacts/real/algorithm2-run/real-main-01/` against its `freeze_manifest.json` |
| `two_tower: not_configured` | Same bundle problem, or the `ml` extra is absent (two-tower still lists as ready if `C0_seed42.pt` loads; it needs torch to run, otherwise B1 is used as default) |
| `retrieval_dense: degraded` | `ml` extra (sentence-transformers / BGE model) not installed; lexical FTS5 retrieval is used instead |
| `llm: not_configured` | No `ANILOGIC_LLM_API_KEY` set; chat works in local evidence-template mode |
| Chat returns HTTP 500 on unusual studio/title names | `rapidfuzz` missing — it is a core dependency, reinstall from `requirements.txt` |
| Catalog looks stale after editing the database | The catalog is an in-memory snapshot loaded at startup — restart the server |
| Port already in use | `PORT=<other>` or kill the previous process; check with `netstat -ano | findstr :8000` |
| Artifact verification failure at startup | A file under `artifacts/` was modified or is missing; restore it (re-clone via Git LFS) or re-run the pipeline |
| `ModuleNotFoundError: app` | Run uvicorn from the `system code` directory, or `pip install -e .` |

---

## 8. Stopping

`Ctrl+C` in the terminal running uvicorn. For Docker: `docker compose down`.
There is no state to flush — sessions live in the SQLite database and the
catalog snapshot is rebuilt on the next start.
