# AniLogic — Startup Guide

How to install, configure, start and verify the system. Requirements are listed
in `SYSTEM_REQUIREMENTS.md`; a map of every file is in `SYSTEM_OVERVIEW.md`.

All commands below are run from the `system code` directory.

From the repository root, use `cd "system code"` (the directory name contains a
space). Before starting, install Git LFS and fetch the system's large files:

```bash
# Run these two commands from the repository root.
git lfs install
git lfs pull --include="system code/**" --exclude=""
```

A small text pointer named `anilogic.sqlite3` is not a database. Confirm that the
real database is hundreds of MB, and leave space for the environment, model
cache and test database copies in addition to the checkout.

---

## 1. Install dependencies

### Option A — uv (recommended, matches `uv.lock` exactly)

```bash
uv sync --extra ml --extra llm --extra dev   # full environment
# or the minimal core only:
uv sync
```

`uv sync` creates `.venv`; it does not activate it in your shell. Use
`uv run --no-sync python ...` below, or activate `.venv` explicitly. Browser
tests also need the matching Chromium build (install it after the `dev` extra):

```bash
uv run --no-sync python -m playwright install chromium
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate          # bash/zsh
pip install -r requirements.txt
pip install --no-deps -e .
# requirements.txt already pins the ml and llm dependencies.
# To run tests as well:
pip install pytest playwright ruff
python -m playwright install chromium
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`.venv\Scripts\activate.bat` in cmd. If activation is restricted, invoke
`.\.venv\Scripts\python.exe` directly instead.

With GNU Make and a POSIX shell, `make setup` installs the full environment and
Chromium; `make setup-base` installs core plus development dependencies. The
Makefile uses `.venv/bin/python` and POSIX environment assignments, so its recipes
are not native PowerShell/cmd commands.

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

Full real-mode collection recommendations also need the frozen serving bundle
in `artifacts/real/algorithm2-run/real-main-01/`. It ships separately from
`Experiments/`; the experiment directory is not required to serve the website.

### Local configuration and DeepSeek

Copy `.env.example` to `.env` and edit the values locally. `.env` is ignored by
Git. The application reads `ANILOGIC_*` entries automatically; exported process
variables override them. For the official DeepSeek API, an example is:

```dotenv
ANILOGIC_DATASET_MODE=real
ANILOGIC_LLM__PROVIDER=openai_compatible
ANILOGIC_LLM__BASE_URL=https://api.deepseek.com/v1
ANILOGIC_LLM__MODEL=deepseek-flash
ANILOGIC_LLM__TIMEOUT_S=60
ANILOGIC_LLM__MAX_OUTPUT_TOKENS=1400
ANILOGIC_LLM_API_KEY=replace-with-your-own-key
```

Use a model returned by your provider's `/models` endpoint. The ModelScope
example in `.env.example` uses a different provider endpoint and its own
credentials. DeepSeek uses the core `httpx` adapter; the `llm` extra is needed
only for the Anthropic SDK. Never put a real API key in a tracked file.

### Cache the dense-query model

Installing the `ml` extra does not download BGE model weights. Startup loads the
query encoder from the local Hugging Face cache only. Download it once:

```bash
uv run --no-sync python -c "from app.retrieval.embeddings import BGEEncoder; print(BGEEncoder().info())"
```

If you set `HF_HOME`, use the same process environment when downloading and
serving. `HF_HOME` is not an `ANILOGIC_*` setting and is not loaded from the
application's `.env` parser. Without the cached model, lexical retrieval still
works and `/status` reports dense retrieval as degraded.

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
ANILOGIC_DATASET_MODE=real uv run --no-sync python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
$env:ANILOGIC_DATASET_MODE = "real"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In cmd, use `set ANILOGIC_DATASET_MODE=real` followed by the same Python command
without the leading `.\`. An activated virtual environment also allows plain
`python -m uvicorn ...`. If port 8000 is occupied, choose another `--port`, for
example 8001, and use it in every URL below.

### Docker

```bash
# POSIX shell: initialize a synthetic world in Docker's named volumes first.
ANILOGIC_DATASET_MODE=synthetic docker compose --profile setup run --rm setup
ANILOGIC_DATASET_MODE=synthetic docker compose up --build
```

The supplied Compose file uses **named volumes** for `data`, `artifacts`, `var`
and the model cache; only `./data/raw` is a host bind mount. It does not expose
the shipped real database or trained artifacts automatically. To run the
shipped real dataset in Docker, supply a separate Compose override with those
host directories mounted, or populate the named volumes first. Its LLM
environment list also needs the chosen base URL and model; passing only the
provider and key is insufficient. The native startup above is the verified
path for the shipped real dataset; Docker requires its own runtime validation.

---

## 4. What startup does

1. Runs Alembic migrations on the mode's database (`var/<mode>/anilogic.sqlite3`).
2. Loads the entire catalog from SQLite into memory (a snapshot — edits to the
   database file while running are not visible until restart).
3. Verifies the activation manifest and hashes of model files it loads against
   the active artifact bundle.
4. Loads the frozen algorithm-2 bundle from
   `artifacts/real/algorithm2-run/real-main-01/` and checks the seed-42 models'
   hashes against its `freeze_manifest.json`.

Startup time depends on disk, CPU, model imports and cache state. Wait for
`serving state loaded` and a health response with `ok: true`; HTTP 200 alone
does not prove that a dataset has been activated.

---

## 5. Verify it is running

```bash
curl http://127.0.0.1:8000/api/v1/system/health
# -> {"ok":true,...}

curl http://127.0.0.1:8000/api/v1/system/status
# capability matrix: catalog, each algorithm, retrieval, llm, ...
```

Then open `http://127.0.0.1:8000/` in a browser:

1. Click **Register** and create an account (synthetic mode has demo accounts
   from `make seed`).
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

### Serving the snapshot versus rebuilding it

The shipped real database already contains the supplementary theme/demographic
rows used by the frozen algorithm-2 bundle, so ordinary startup is independent
of the experiment directory. However, `configs/base.yaml` still sets
`recommendation.collection.supplement` to
`algorithm1_benchmark/data/catalog.jsonl.gz`, a path absent from this system
directory after the repository split. Before re-activating or rebuilding the
real dataset, provide a system-owned copy of the supplementary data and update
that setting. Otherwise activation records the supplement as missing and may
make the frozen collection bundle incompatible. Rebuilding is not needed to
start the shipped system, and it was not part of the local serving audit.

The full system test suite also imports the published algorithm-1 benchmark
from `../Experiments/` for its 1,000-query parity check. Keep that sibling folder
when running all tests; serving the website does not need it.

Pipeline stages (only needed when rebuilding data from raw CSVs) are dispatched
through `uv run --no-sync python -m pipelines.cli --help` (preprocess -> split -> train ->
build-index -> activate).

On Windows, the test commands corresponding to `make test` and `make e2e` are:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -ra
.\.venv\Scripts\python.exe -m pytest tests/e2e -m e2e -ra
```

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `collection_scorer: degraded` or `A0` in `/status` | Frozen algorithm-2 bundle missing/hash mismatch — check `artifacts/real/algorithm2-run/real-main-01/` against its `freeze_manifest.json` |
| `two_tower: not_configured` | Same bundle problem, or the `ml` extra is absent (two-tower still lists as ready if `C0_seed42.pt` loads; it needs torch to run, otherwise B1 is used as default) |
| `retrieval_dense: degraded` | `ml` extra (sentence-transformers / BGE model) not installed; lexical FTS5 retrieval is used instead |
| `llm: not_configured` | No `ANILOGIC_LLM_API_KEY` set; chat works in local evidence-template mode |
| `llm: ready` but answers use `local_template` | The status checks configuration, not API reachability. Inspect the response's `meta.llm` and `warnings` for authentication, timeout, JSON or validation failures; some intents deliberately use local templates |
| `Readme file does not exist: README.md` during install | `pyproject.toml` must reference the existing `SYSTEM_OVERVIEW.md`; container builds must copy that file too |
| Chromium executable missing in e2e tests | Run `python -m playwright install chromium` with the same environment and `PLAYWRIGHT_BROWSERS_PATH` as the tests |
| BGE download exits in the Xet component | Retry with `HF_HUB_DISABLE_XET=1` in the process environment to use standard HTTPS |
| Disk full during installation or tests | Allow space for ML wheels, browser extraction, model caches and temporary database copies; move the task's environment/cache to a drive with space |
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
