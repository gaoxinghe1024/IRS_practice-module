# Algorithm 2 experiment package: comparing five options for collection recommendation, pre-registered selection and system integration

This mirrors the recommendation flow on the collection page: **collection content H → preference profile → fixed rule layer → swappable scorer → Top-8**, comparing the
**A0 original manual formula, A1 tuned-weight formula, B0 two-feature GBDT, B1 eight-feature GBDT and C0 content two-tower** from the slides. All options are compared on the
same real catalogue (14,182 titles), the same user split, the same candidates and rules, and the same evaluator; the selection rule was written and frozen before any training.

Start with the [formal run report](results/real-main-01/final_report.md), the [English summary table](results/real-main-01/summary_en.md),
the [failure cases](results/real-main-01/case_analysis.md), the [paired comparisons](results/real-main-01/paired_comparisons.csv),
the [latency](results/real-main-01/latency.csv), the [integration checks](results/real-main-01/integration_report.json) and the
[test output](results/tests.txt).

Labels are **proxy labels**: the set of titles a user rated ≥ 8 is treated as the "collection", one of them is hidden, and each option is asked to put it in the Top-8.
The application database holds no real collections yet, so every conclusion is a "high-rating preference set proxy evaluation", not collection behaviour itself.

Formal run `real-main-01`: 24,000 users, a test set of 3,000 main-group users, full-catalogue ranking, rule R0, user-level macro metrics.

| Model | HR@8 | NDCG@8 | AUC | ΔHR@8 vs A0 (pp, 95% interval) | ΔHR@8 vs A1 (pp, 95% interval) |
|---|---|---|---|---|---|
| A0 original manual formula | 0.0210 | 0.0107 | 0.8699 | — | −1.10 [−1.60, −0.57] |
| A1 validation-tuned formula (w = 0.05) | 0.0320 | 0.0179 | 0.9082 | +1.10 [+0.57, +1.60] | — |
| B0 LightGBM two features (seed 42) | 0.0543 | 0.0316 | 0.9213 | +3.33 [+2.50, +4.17] | +2.23 [+1.53, +2.97] |
| **B1 LightGBM eight features (seed 42, deployed)** | **0.1317** | **0.0723** | **0.9527** | **+11.07 [+9.83, +12.30]** | **+9.97 [+8.73, +11.17]** |
| C0 content two-tower (seed 42) | 0.1083 | 0.0621 | 0.9640 | +8.73 [+7.53, +9.93] | +7.63 [+6.53, +8.77] |

- Mean over the three training seeds: B0 0.0532, B1 0.1320, C0 0.1133 (standard deviation 0.0008 / 0.0010 / 0.0037).
- Key comparisons: A1 − A0 = +1.10 [+0.57, +1.60] (a small gain from tuning the weight); B0 − A1 = +2.12 [+1.48, +2.84] (with the same two features, learning the
  scoring function helps); **B1 − B0 = +7.88 [+6.78, +8.99]** (the six profile features bring the main gain); C0 − B1 = −1.87
  [−3.03, −0.69] (the two-tower is below the eight-feature GBDT; C0 has the highest AUC but a worse Top-8 than B1).
- **Selection conclusion**: B1 was pre-selected on the validation set (HR@8 0.1301) and passed every test confirmation check (AUC ≥ 0.65, at least 2 percentage points
  above A0, 0 rule violations, and the lower bound of the paired interval against both A0 and A1 above 0). The default scorer for the site's real mode is B1; synthetic mode stays on A0.
- The smoke run `real-smoke-01` (150 test users) pre-selected C0, but its interval against A1 contains 0, so the protocol keeps A1: conclusions are
  unstable at that sample size, and only the main run counts as formal.
- Serving latency (single thread on this machine, service-layer P50 for one request over the full catalogue): about 120 ms for B1, about 7 ms for the formulas, about 1 ms for C0.

## 1. Running it directly

The commands below are run from the **repository root**. Unlike Algorithm 1's standalone package, this experiment calls the site code directly (catalogue, rule layer, features and scorers all live in
`app/`), so the offline evaluation and the site follow the same code path; running it needs the repository's Python environment and the activated real data.

```bash
uv sync --extra ml                 # Python 3.12 environment (LightGBM 4.7, PyTorch)
brew install libomp                # macOS only: the OpenMP runtime LightGBM depends on

.venv/bin/python algorithm2_benchmark/verify_delivery.py      # check the delivered files first (SHA-256 manifest)
.venv/bin/python algorithm2_benchmark/run_checks.py           # tests, actual output written to results/tests.txt

.venv/bin/python -m algorithm2_benchmark report --run real-main-01              # regenerate the report from existing records (seconds)
.venv/bin/python -m algorithm2_benchmark verify-integration --run real-main-01  # API / offline consistency (about 1 minute)
.venv/bin/python -m algorithm2_benchmark benchmark --run real-main-01           # latency (about 5 minutes)

.venv/bin/python -m algorithm2_benchmark run-all --run my-smoke --level smoke   # a new full run (about 5 minutes)
.venv/bin/python -m algorithm2_benchmark run-all --run my-main --level main     # formal scale (about 3 hours on this machine)
```

The Makefile exposes the same entry points: `make alg2-run RUN=... LEVEL=...`, `make alg2-stage STAGE=... RUN=...`,
`make alg2-checks`, `make alg2-verify`, `make alg2-fixture` (checks the pipeline on generated small data, not an effectiveness result).

The stages are `audit`, `prepare`, `split`, `tune`, `train`, `freeze`, `evaluate-test`, `benchmark`,
`verify-integration`, `report`; `run-all` executes them in order, skips stages that have already completed, and runs each stage in its own process. A new run needs
`--level fixture|smoke|main`; an existing run keeps its own `config.json`; `--threads N` sets the LightGBM / PyTorch thread count.

The delivery manifest records the originally delivered files; re-running stages rewrites the run metadata, reports and test output, so verify before re-running. Once the new delivery is confirmed,
update the manifest with `verify_delivery.py --write`.

## 2. File layout

| Path | Purpose |
|---|---|
| `pipeline/cli.py`, `config.py`, `common.py` | command-line entry point and stage orchestration, run configuration (every value that affects a result), hashing / logging / run metadata |
| `pipeline/audit.py` | repository and data facts (backs `docs/repo_audit.md`) |
| `pipeline/data.py` | read and validate the processed real data, clean ratings, build proxy positives; the fixture small data |
| `pipeline/split.py` | user-level split, hidden targets, R1 keywords, 1:4 weak-negative sampling, the pre-registered `selection_protocol.json` |
| `pipeline/train.py` | A1 weight tuning, B0 / B1 LightGBM and C0 two-tower tuning plus the final three-seed training (LightGBM and PyTorch in separate processes) |
| `pipeline/select.py` | validation pre-selection, freeze (model hashes, consistency check between native LightGBM and the serving evaluator), test confirmation |
| `pipeline/evaluate.py`, `metrics.py`, `final_eval.py` | full-catalogue evaluator, metric definitions, test-set evaluation, subgroups and paired comparisons |
| `pipeline/benchmark.py` | three-layer latency (scoring only / service / API) and load cost |
| `pipeline/integration.py`, `webapp.py` | run the site against a copy of the real database and check the integration (14 checks) |
| `pipeline/report.py` | report, English summary table, failure cases, reproduction notes |
| `results/real-main-01/` | formal run: configuration, data / split / selection protocol / freeze manifest, `models/` (9 models), `predictions.parquet` (per query × rule × model), metrics, subgroups, paired comparisons, latency, integration checks, reports |
| `results/real-smoke-01/` | smoke run (the same records at a small scale, pipeline check only) |
| `results/tests.txt` | actual output of the most recent `run_checks.py` |
| `docs/repo_audit.md` | the repository audit taken before any change |
| `tests/` | the experiment's own tests: metric definitions, full fixture flow (split and leakage, real fitting, candidate consistency, offline = serving, no retuning after freeze, records complete) |
| `run_checks.py`, `verify_delivery.py`, `DELIVERY_MANIFEST.json` | test entry point, delivery verification and manifest |

Each run directory's `work/` holds regenerable intermediates (positives, features, per-process predictions, logs) and is not committed; without
`work/`, the report and integration stages read the published `query_manifest.parquet` and `sampled_pairs.parquet` from the run directory.

**The code the site runs on is not in this folder**: `app/recommendation/algorithm2/` (catalogue, profiles, eight features, scorers, NumPy evaluation of the LightGBM
text model, model-bundle verification, serving), the rule layer `app/recommendation/constraints.py`, and the collection endpoint
`app/services/recommend.py`. The tests guarding site behaviour are `tests/unit/test_collection_algorithm2.py` and
`tests/integration/test_collection_algorithm2.py` (`run_checks.py` runs both).

Both runs were completed before this folder was reorganised: the `evaluation/algorithm2/` paths in `run_metadata.json`, `audit_facts.json` and
`freeze_manifest.json` are where the code lived at the time and are kept as they are as a historical record.

## 3. Options and metric conventions

| ID | Scoring |
|---|---|
| A0 | `rating_norm + 0.15 × hits` (hits = the number of candidate genre / theme tags that fall inside the top-10 namespace tags of H) |
| A1 | `rating_norm + w × hits`, with w chosen on the validation set only, from [0.05, 0.10, 0.15, 0.20, 0.30] |
| B0 | LightGBM binary classification (objective = binary) over `rating_norm` and `hits`, ranked by raw score |
| B1 | as above, over eight fixed features: `rating_norm`, `hits`, genre / theme / demographic / studio affinity, episode-length closeness, and tag similarity to the most similar title in the collection |
| C0 | PyTorch content two-tower: a collection tower and an anime tower, each 128 → 64 → 32/64, over multi-hot tags, studio, numeric features and a missing-value mask; dot-product logits + BCEWithLogitsLoss, no in-batch negatives |

- Profile frequency = count / |H|; missing features stay NaN and are never invented. B0 / B1 / C0 share the same 1:4 training sample set (`negative_seed = 42`).
- Metrics: HR@8, NDCG@8 (single positive, IDCG = 1), MRR@8, AUC (ties count 0.5; null when there are fewer than 2 candidates). Values are averaged per user and then
  across users; ties are broken by ascending MAL id but that ordering does not enter AUC.
- Primary set (pre-registered): rule R0, main group (visible history ≥ 5), more than 8 candidates, target satisfies the rule.
- Rule scenarios: R0 has no filter; R1 takes the most common genre in H as the keyword; R2 requires public score ≥ 7.0 and episodes ≤ 26 (an unknown episode count does not satisfy it);
  R3 = R1 + R2. Strict mode: a target that does not satisfy the rule is not inserted back, and end-to-end results are reported separately.
- Test confirmation (against A0): AUC ≥ 0.65, HR@8 gain ≥ 2 percentage points, 0 rule violations, and the lower bound of the 95% user-paired bootstrap interval (2,000 resamples)
  above 0; a learned model must also beat A1 with a lower bound above 0, otherwise A1 is kept.

## 4. Experiment protocol and freeze protection

- Data: Kaggle MyAnimeList 2023 snapshot (processed version `real-89bfdea00b`; `prepare` checks file hashes against the manifest), MAL id only;
  catalogue = servable (Hentai / Erotica / Rx excluded, Ecchi and R+ kept) and having a public score; theme / demographic merged by MAL id from
  `algorithm1_benchmark/data/catalog.jsonl.gz`.
- Split: user-level 70 / 15 / 15, `split_seed = 42`, stratified by preference-set size; main group (≥ 6) and sparse group (2–5) reported separately; validation /
  test hide one title per user, training up to three; the target never enters the profile input.
- Selection: `selection_protocol.json` is written before any training; parameters and pre-selection use the validation set only; B / C use training seed [42, 2026, 7] with
  deploy seed 42; after `freeze`, `tune` / `train` refuse to run and re-running `evaluate-test` only reproduces the same result.
- Process isolation: on macOS, LightGBM (Homebrew libomp) and PyTorch (bundled libomp) deadlock or crash in the same process, so training and
  evaluation put them in separate subprocesses; the site evaluates the LightGBM text model directly in NumPy (tree by tree over the bit masks, the raw score matching native
  LightGBM exactly, checked on 560,000 rows at `freeze` time with a maximum difference of 0).

## 5. Wiring it into the site

`configs/real.yaml`:

```yaml
recommendation:
  collection:
    scorer: B1
    algorithm2_run: algorithm2_benchmark/results/real-main-01
```

At start-up the service verifies the model file SHA-256, the catalogue hash, the feature version and the two-tower encoder; any mismatch falls back to formula A0, the status page shows
`collection_scorer: degraded`, and the user's filters are still enforced strictly. When a strict query returns nothing it only offers relaxation, and after the user chooses, the response carries
`relaxed`, `relaxed_rules` and `strict_result_count`, with the page saying which condition was relaxed. The endpoint's `scorer` field is for experiments and
operations only; the page never sends it. `verify-integration` checks against a database copy: the API Top-8 matches the offline prediction 1,000 / 1,000,
a collection change changes the output, a short strict list is not padded, an empty result is not silently relaxed, recommended ids open their detail pages, and a missing or modified model bundle falls back to the formulas.

## 6. Limitations

- A proxy label is not a collection; unrated titles are weak negatives. There are no real collections and no online experiment, so the actual effect on the site is not yet validated; once
  real collection data exists, the same pipeline can re-evaluate it.
- Static snapshot and held-out users: what is evaluated is "can other high-rated titles from the same period be recovered", not a prediction of future behaviour.
- Targets are mostly popular high-rated titles, so there is a popularity bias; obscure targets rarely reach the top 8.
- C0 can encode new titles with no interactions, but there was no item-disjoint experiment, so we cannot claim better cold start.
- The LightGBM seed only affects the row sampling of histogram binning (the grid has no subsampling), so differences between seeds do not represent the full training randomness.
- Latency is an observation on a single laptop, and the project has no fixed SLO.
