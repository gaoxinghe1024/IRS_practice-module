# AniLogic developer commands. `python -m ...` is used everywhere so the commands work even when an
# editable install's .pth file is hidden (macOS iCloud folders set the UF_HIDDEN flag; see docs/operations.md).
PY ?= .venv/bin/python
MODE ?= synthetic
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: setup setup-base seed build-resources serve serve-real test test-unit test-integration test-real e2e eval-smoke \
        validate-real preprocess-real split-real train build-index activate-real evaluate export-evaluation \
        enrich-jikan migrate-legacy status lint clean-synthetic requirements apply-tuning alg2-run alg2-stage \
        alg2-fixture alg2-checks alg2-verify

setup:            ## full environment: base + ml (torch, sentence-transformers) + llm (anthropic) + dev
	uv sync --extra ml --extra llm --extra dev
	-chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true
	$(PY) -m playwright install chromium

setup-base:       ## minimal environment (no neural extras: LSA retrieval, no NCF / BGE / two-tower)
	uv sync --extra dev
	-chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true

seed:             ## synthetic world: generate -> preprocess -> split -> train -> index -> activate -> demo accounts
	$(PY) -m pipelines.cli build-all --mode synthetic

build-resources:  ## (re)train models and rebuild indexes for MODE, then activate
	$(PY) -m pipelines.cli train --mode $(MODE)
	$(PY) -m pipelines.cli build-index --mode $(MODE)
	$(PY) -m pipelines.cli activate --mode $(MODE)

serve:            ## run the app (MODE=synthetic|real)
	ANILOGIC_DATASET_MODE=$(MODE) $(PY) -m uvicorn app.main:app --host $(HOST) --port $(PORT)

serve-real:
	$(MAKE) serve MODE=real

test:             ## unit + integration tests (builds a miniature synthetic world in a temp dir)
	$(PY) -m pytest tests/unit tests/integration

test-unit:
	$(PY) -m pytest tests/unit

test-integration:
	$(PY) -m pytest tests/integration

test-real:        ## real-data and model checks only (need an activated real version / the cached BGE model)
	$(PY) -m pytest tests -m "real or ml"

e2e:              ## browser journeys (Playwright, Chromium)
	$(PY) -m pytest tests/e2e -m e2e

eval-smoke:       ## quick run of every experiment on small samples (checks the harness; NOT a quality evaluation)
	$(PY) -m pipelines.cli evaluate --mode $(MODE) --experiment all --sample-users 40 --seeds 1 --tag smoke

validate-real:
	$(PY) -m pipelines.cli validate-real

preprocess-real:
	$(PY) -m pipelines.cli preprocess --mode real

split-real:
	$(PY) -m pipelines.cli split --mode real

train:
	$(PY) -m pipelines.cli train --mode $(MODE)

build-index:
	$(PY) -m pipelines.cli build-index --mode $(MODE)

activate-real:
	$(PY) -m pipelines.cli activate --mode real

evaluate:         ## full evaluation for MODE
	$(PY) -m pipelines.cli evaluate --mode $(MODE) --experiment all

export-evaluation:
	$(PY) -m pipelines.cli export-evaluation --mode $(MODE)

apply-tuning:     ## copy the validation-tuned ICF parameters of the latest homepage run into configs/$(MODE).yaml
	$(PY) -m pipelines.apply_tuning --mode $(MODE)

enrich-jikan:     ## optional offline Jikan enrichment (real mode; resumable; conservative rate)
	$(PY) -m pipelines.cli enrich-jikan --top-popular 400

migrate-legacy:   ## dry run of the legacy account migration (add ARGS=--apply to write)
	$(PY) -m pipelines.cli migrate-legacy --mode $(MODE) $(ARGS)

status:
	$(PY) -m pipelines.cli status --mode $(MODE)

lint:
	$(PY) -m ruff check app pipelines evaluation tests algorithm2_benchmark

# Algorithm 2 (collection page) experiments, all in algorithm2_benchmark/ (see its README.md). RUN names the run
# directory under algorithm2_benchmark/results/, LEVEL is fixture | smoke | main (needed only when the run is created).
RUN ?= real-main-01
LEVEL ?= main
STAGE ?= report

alg2-run:         ## every algorithm-2 stage in order for RUN (finished stages are skipped)
	HF_HUB_OFFLINE=1 $(PY) -m algorithm2_benchmark run-all --run $(RUN) --level $(LEVEL)

alg2-stage:       ## one algorithm-2 stage: make alg2-stage STAGE=benchmark RUN=real-main-01
	HF_HUB_OFFLINE=1 $(PY) -m algorithm2_benchmark $(STAGE) --run $(RUN)

alg2-fixture:     ## the algorithm-2 pipeline on generated fixture data (logic check only, not a quality result)
	HF_HUB_OFFLINE=1 $(PY) -m algorithm2_benchmark run-all --run fixture-check --level fixture \
		--root $${TMPDIR:-/tmp}/alg2-fixture-runs

alg2-checks:      ## algorithm-2 experiment tests; the output is saved to algorithm2_benchmark/results/tests.txt
	HF_HUB_OFFLINE=1 $(PY) algorithm2_benchmark/run_checks.py

alg2-verify:      ## check the delivered algorithm-2 records against algorithm2_benchmark/DELIVERY_MANIFEST.json
	$(PY) algorithm2_benchmark/verify_delivery.py

requirements:     ## regenerate the pinned requirements.txt (pip users) from uv.lock
	uv export --frozen --no-hashes --no-dev --extra ml --extra llm --no-emit-project -o requirements.txt

clean-synthetic:  ## remove generated synthetic data, artifacts and database
	rm -rf data/synthetic data/processed/synthetic artifacts/synthetic var/synthetic
