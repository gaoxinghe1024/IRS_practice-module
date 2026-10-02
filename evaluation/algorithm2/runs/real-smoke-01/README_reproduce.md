# Reproducing run `real-smoke-01`

Level: `smoke`. All commands run from the repository root.

## Environment

```bash
uv sync --extra ml          # Python 3.12 environment with LightGBM 4.7.0 and PyTorch
brew install libomp         # macOS only: the OpenMP runtime LightGBM links against
```

LightGBM and PyTorch are never loaded into the same process (their OpenMP runtimes deadlock on macOS); the stages
that need both start isolated sub-processes automatically.

## Data

- Processed real dataset `data/processed/real/real-89bfdea00b`
  (Kaggle MyAnimeList 2023 snapshot, Git LFS), verified against its manifest by `prepare`.
- Theme / demographic supplement `algorithm1_benchmark/data/catalog.jsonl.gz` (joined by MAL id).

## Commands

```bash
# every stage in order (finished stages are skipped; a new run needs --level)
.venv/bin/python -m evaluation.algorithm2.cli run-all --run real-smoke-01 --level smoke

# or one stage at a time
.venv/bin/python -m evaluation.algorithm2.cli audit              --run real-smoke-01 --level smoke
.venv/bin/python -m evaluation.algorithm2.cli prepare            --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli split              --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli tune               --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli train              --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli freeze             --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli evaluate-test      --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli benchmark          --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli verify-integration --run real-smoke-01
.venv/bin/python -m evaluation.algorithm2.cli report             --run real-smoke-01
```

Options: `--threads N` (LightGBM / PyTorch threads, fixed at run creation), `--root DIR` (runs directory).
After `freeze`, `tune` and `train` refuse to run; `evaluate-test` only reproduces the frozen evaluation.
The Makefile wraps the same commands (`make alg2-run RUN=real-smoke-01 LEVEL=smoke`).

## Outputs

`evaluation/algorithm2/runs/real-smoke-01/` holds the committed records (manifests, metrics, models, reports); `work/` (git-ignored) holds the
regenerable intermediate files (positives, features, per-pass predictions, logs).

## Serving the result

Set in `configs/real.yaml`:

```yaml
recommendation:
  collection:
    scorer: A1
    algorithm2_run: evaluation/algorithm2/runs/real-smoke-01
```

The web app verifies the model hashes, catalogue hash, feature version and encoder before loading; on any mismatch
it serves the formula and reports `collection_scorer: degraded` on the status page.
