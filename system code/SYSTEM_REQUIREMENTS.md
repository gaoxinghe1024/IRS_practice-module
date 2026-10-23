# AniLogic — System Requirements

Everything the machine needs before the system can be installed and started.
For the actual startup steps see `STARTUP_GUIDE.md`; for a map of the code see
`SYSTEM_OVERVIEW.md`.

---

## 1. Software requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | **>= 3.11, < 3.14** | 3.12 is what the lockfile and Docker image target |
| [uv](https://docs.astral.sh/uv/) | any recent | Recommended installer; `uv.lock` is the source of truth. pip + `requirements.txt` also works |
| Make | any | Optional but convenient; every command exists as a plain Python/uv invocation too |
| Docker | 24+ | Optional, only for the containerized deployment |
| Git + Git LFS | LFS 3.x | Required when cloning the repository — all datasets, models and databases are LFS-tracked |

Operating system: Windows, macOS or Linux. Nothing in the codebase is OS-specific
(paths are handled with `pathlib`; the frozen-model loading is byte-exact on
Windows too thanks to `-text` rules in `.gitattributes`).

---

## 2. Python dependencies

### Core (always required)

Installed by `uv sync` or `pip install -r requirements.txt`:

fastapi, uvicorn[standard], pydantic, pyyaml, jinja2, python-multipart,
sqlalchemy, alembic, argon2-cffi, httpx, numpy, scipy, pandas, pyarrow,
scikit-learn, **rapidfuzz**, pillow

`rapidfuzz` is a core dependency: the chat constraint extractor uses it for
fuzzy matching of titles/studios/genres and has no fallback. Without it, chat
messages containing out-of-vocabulary entity names return HTTP 500.

### Optional extras (pyproject `[project.optional-dependencies]`)

| Extra | Packages | What it unlocks |
|---|---|---|
| `ml` | torch, sentence-transformers, lightgbm | Dense BGE retrieval, NCF recall, two-tower model training. Without it the system still serves all four algorithms (dense retrieval degrades to lexical FTS5; algorithm 2 uses the shipped frozen LightGBM trees which are parsed in pure numpy) |
| `llm` | anthropic | LLM-backed chat answers. Without it chat runs in the local evidence-template mode |
| `dev` | pytest, playwright, ... | Test suites and e2e browser tests |

### LLM (chat) — optional

No API key is needed to run the system. Without one, algorithm 4 answers from
retrieved evidence via local templates. To enable an LLM, set
`ANILOGIC_LLM__PROVIDER`, `ANILOGIC_LLM__BASE_URL`, `ANILOGIC_LLM__MODEL` and
`ANILOGIC_LLM_API_KEY` (see `.env.example` for supported providers).

---

## 3. Data requirements

The repository ships with everything needed for **real mode**:

| Item | Size | Required? |
|---|---|---|
| `data/raw/mal_2023/` (Kaggle Anime Dataset 2023) | ~1.5 GB with all data dirs | Only to re-run the ingest pipeline |
| `var/real/anilogic.sqlite3` | ~350 MB | **Yes for real mode** — the business database (24,905 anime, tags, ratings, embeddings metadata) |
| `artifacts/real/real-034ebd6a74/` | ~278 MB | **Yes for real mode** — trained models/indexes; startup verifies sha256 per `manifest.json` |
| `artifacts/real/algorithm2-run/real-main-01/` | ~4 MB | **Yes for full algorithm 2** — frozen scorer bundle; without it algorithm 2 degrades to the manual A0 formula |
| `data/synthetic/` + `var/synthetic/` + `artifacts/synthetic/` | small | Only for synthetic demo mode; regenerate with `make seed` if absent |

If any artifact file is missing or its hash mismatches the manifest, startup
either fails verification or logs the capability as `degraded` — check
`/api/v1/system/status` after starting.

---

## 4. Hardware / resource requirements

| Resource | Minimum | Comfortable |
|---|---|---|
| RAM | 4 GB | 8 GB (the catalog snapshot + indexes take ~1.5-2 GB; BGE inference needs more) |
| Disk | ~2.5 GB for the repo as shipped | +2 GB if you rebuild models |
| CPU | 4 cores | 8 cores (uvicorn thread pool scales with `app.cpu_workers`) |
| GPU | none | Optional; only relevant for retraining NCF/two-tower or BGE encoding |

Startup loads the whole catalog into memory and runs Alembic migrations, so the
first request after launch takes a few seconds on real mode (a fresh process is
ready in roughly 10-20 s on a laptop).

---

## 5. Network & port requirements

| Need | Detail |
|---|---|
| Free TCP port | Default **8000** (`app.port`), configurable via Makefile `PORT=` or env |
| Bind address | Default `127.0.0.1`; use `HOST=0.0.0.0` for LAN access |
| Outbound internet | Optional: cover-image proxying, Jikan enrichment, LLM APIs, Hugging Face model downloads. Fully offline once artifacts are present |
| HTTPS | Off by default; set `ANILOGIC_APP__SECURE_COOKIES=true` when terminating TLS in front |

---

## 6. Quick pre-flight check

```bash
python --version          # must be 3.11.x - 3.13.x
git lfs version           # if cloning fresh
uv --version              # optional but recommended
ls var/real               # anilogic.sqlite3 present?
ls artifacts/real         # real-034ebd6a74/ and algorithm2-run/real-main-01/ present?
```

All green → continue with `STARTUP_GUIDE.md`.
