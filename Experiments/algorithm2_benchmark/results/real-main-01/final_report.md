# Algorithm 2 (collection recommendation): effectiveness comparison, model selection and system integration report — completed

Run `real-main-01` · level **main** · generated at 2026-10-02T08:46:03Z. Every number in this report comes from a record file in this run directory (see section 9); anything that did not run is written as NOT_RUN.

## 1. State of the original system, scope of this work and what was added

- Code starting point: branch `yyc-algorithm2-selection`, commit `3cda03a90a` (identical to origin/main). The repository audit is in `algorithm2_benchmark/docs/repo_audit.md`.
- Original state: the collection page used `collection_profile_v1` (normalised profile matching, α = 0.7) as the default scorer, with `manual_slides_v0` (the slides formula, counting hits by presence) kept only as a reference; there was no trained collection model, no fixed user split, no pre-registered selection rule and no offline / online consistency check.
- Added in this work: `app/recommendation/algorithm2/` (unified catalogue, profiles, eight features, five scorers, a NumPy reader for the LightGBM text model, model-bundle verification and serving), `algorithm2_benchmark/` (a ten-stage command-line experiment pipeline, run records and reports), the collection endpoint now calling the same code path, the strict rules (with the new `pool.scored`) and an explicit-relaxation response contract, and every record behind this report.

## 2. Data source, labels, cleaning and scale

- Data: Kaggle MyAnimeList 2023 snapshot, processed version `real-89bfdea00b` (raw-file SHA-256 in data_manifest.json). Only the MAL id is used as the item identity.
- Recommendation catalogue: 14,182 titles = servable (Hentai / Erotica / Rx excluded; Ecchi and R+ kept) and having a public score. The snapshot has 24,905 rows. Catalogue hash `24c831926b61`.
- Field coverage: genre 87.6%, theme 68.9%, demographic 37.2%, studio 76.2%, episodes 99.6%, score 100.0%. Theme / demographic were supplemented from the teammate-provided `algorithm1_benchmark/data/catalog.jsonl.gz` joined by MAL id (union; 9,760 theme and 5,260 demographic values added); missing fields are never invented, and an unknown episode count does not satisfy the episode-cap rule.
- Interaction cleaning: 23,963,380 ratings read, 23,954,329 kept inside the catalogue (9,051 outside, all of them lacking a public score); no invalid scores and no conflicting duplicates.
- Label: a **proxy label** — the titles a user rated ≥ 8 form that user's high-rating preference set (13,580,272 rows, 264,020 users). The application database holds no real collections, so every effectiveness conclusion here is an evaluation on that high-rating preference set used as a proxy, not collection behaviour itself; unrated titles are weak negatives.
- Preference-set size distribution (users): 1-1: 23,055, 2-5: 38,763, 6-9: 22,955, 10-19: 35,764, 20-49: 55,889, 50-99: 43,083, 100+: 44,511.

## 3. Splits, leave-one-out, rule scenarios, candidates and leakage control

- User-level split 70 / 15 / 15, `split_seed = 42`, stratified by preference-set size; the main group (set ≥ 6, i.e. visible history ≥ 5) and the sparse group (2–5) are reported separately. Full split: main / test 3,000, main / train 14,000, main / val 3,000, sparse / test 600, sparse / train 2,800, sparse / val 600.
- This level uses 24,000 users (nested in the full split: True). Users shared between partitions: {'train&val': 0, 'train&test': 0, 'val&test': 0}.
- Queries: one hidden target per user for validation / test, up to 3 targets per user for training; 56,676 queries in total ({'main': {'test': 3000, 'train': 42000, 'val': 3000}, 'sparse': {'test': 600, 'train': 7476, 'val': 600}}). The hidden target never enters the profile input H.
- Training samples: 1 positive + 4 weak negatives per training query (deterministic sampling inside the same candidate set, `negative_seed = 42`, excluding all known positives of that user), 265,380 pairs in total (247,380 training pairs; the 18,000 pairs of validation users are used only for the early-stopping logloss); the negative shortfall occurred 0 times. B0 / B1 / C0 share the same sample set.
- Rule scenarios: R0 has no filter; R1 takes the most common genre in H as the keyword (ties broken by name), and the keyword is matched as an AND substring against genre or title; R2 requires public score ≥ 7.0 and episodes ≤ 26 (an unknown episode count does not satisfy it); R3 = R1 + R2. All of them are strict: a target that does not satisfy the rule is not inserted back, is recorded as target_not_eligible, and end-to-end results are reported separately.
- Candidates: the **full catalogue** passing the rule is ranked (no candidate sampling), excluding H itself. Ties are broken by ascending MAL id; AUC gives 0.5 to ties and is unaffected by the id ordering; AUC is null when there are fewer than 2 candidates.
- Primary metric (pre-registered): user-macro HR@8 on R0, main group, more than 8 candidates, target eligible; NDCG@8, MRR@8 and AUC are reported in the same table.

## 4. Implementation, parameters and budget of the five experiment groups

- **A0** = rating_norm + 0.15 × hits (hits = the number of candidate genre/theme tags that fall inside the top-10 namespace tags of H).
- **A1**: hits weights [0.05, 0.1, 0.15, 0.2, 0.3] compared on the validation set, selected **0.05** (validation HR@8 0.0267); the selected value is the minimum of the pre-registered grid, which suggests the data may prefer a weight outside the grid, but the grid was not extended after the results were seen.
- **B0**: native LightGBM `lgb.train` (objective = binary, ranked by raw score), features rating_norm, hits; 10 pre-generated deterministic trials (grid {'num_leaves': [15, 31], 'max_depth': [5, 8], 'learning_rate': [0.03, 0.05], 'min_child_samples': [50, 100], 'reg_lambda': [1.0, 5.0]}), capped at 500 trees with early stopping after 50 rounds (logloss on validation), trial 0 selected by validation HR@8 over the full candidate set: {'learning_rate': 0.05, 'max_depth': 8, 'min_child_samples': 100, 'num_leaves': 31, 'reg_lambda': 5.0} (validation HR@8 0.0573).
- **B1**: native LightGBM `lgb.train` (objective = binary, ranked by raw score), features rating_norm, hits, genre_affinity, theme_affinity, demographic_affinity, studio_affinity, length_affinity, max_tag_similarity; 10 pre-generated deterministic trials (grid {'num_leaves': [15, 31], 'max_depth': [5, 8], 'learning_rate': [0.03, 0.05], 'min_child_samples': [50, 100], 'reg_lambda': [1.0, 5.0]}), capped at 500 trees with early stopping after 50 rounds (logloss on validation), trial 0 selected by validation HR@8 over the full candidate set: {'learning_rate': 0.05, 'max_depth': 8, 'min_child_samples': 100, 'num_leaves': 31, 'reg_lambda': 5.0} (validation HR@8 0.1320).
- **C0**: PyTorch content two-tower, [128, 64] hidden units per side; inputs are multi-hot vectors over genre / theme / demographic / studio (studios with at least 5 appearances, the rest mapped to UNK) plus numeric features and a missing-value mask (anime tower 378 dims, collection tower 380 dims), dot-product logits + BCEWithLogitsLoss, no in-batch negatives; 10 trials, trial 0 selected: {'dropout': 0.0, 'learning_rate': 0.0003, 'out_dim': 64, 'weight_decay': 1e-05} (validation HR@8 0.1053), at most 30 epochs, patience 3, on CPU.
- The final configuration is trained once per training seed [42, 2026, 7] (9 models in total, see models_manifest.json). Note: the pre-registered LightGBM grid has no bagging / feature_fraction subsampling, so the seed only acts through bin-construction sampling; there are 247,380 training pairs, above LightGBM's histogram bin-construction sample cap of 200,000 (bin_construct_sample_cnt), so the seed decides which rows take part in binning and the bin boundaries and trees of the three seeds differ slightly (checked on the model files: B0 different, B1 different); differences between seeds reflect only this source of randomness.
- Tuning cost (B0 10 trials, B1 10 trials, C0 10 trials, total): B0 fit 52 s + full-candidate validation scoring 2,056 s; B1 fit 53 s + full-candidate validation scoring 2,447 s; C0 fit 380 s (validation scoring not timed separately). Validation scoring over the full catalogue is the main cost.
- Comparison logic: B0 is the two-feature control; B1 and C0 use metadata from the same source but different representations, so C0 vs B1 is not 'the same input with only the model swapped'.
- B1 feature importance (gain share, deploy seed): rating_norm 65.9%, studio_affinity 11.0%, hits 6.5%, length_affinity 5.2%, max_tag_similarity 4.2%, demographic_affinity 2.9%, theme_affinity 2.6%, genre_affinity 1.7%. Importance is not causal contribution; a feature's effect is judged by the B1 − B0 comparison.

## 5. Test results

Test set: 3,600 users and 3,600 queries. The main table covers the pre-registered primary set (R0, main group, >8 candidates, target eligible; user-macro).

### 5.1 Main table

| Model | Users | HR@8 | NDCG@8 | MRR@8 | AUC | ΔHR@8 vs A0 (pp, 95% CI) | ΔHR@8 vs A1 (pp, 95% CI) | Rule violations |
|---|---|---|---|---|---|---|---|---|
| A0 | 3000 | 0.0210 | 0.0107 | 0.0075 | 0.8699 | — | -1.10 [-1.60, -0.57] | 0 |
| A1 | 3000 | 0.0320 | 0.0179 | 0.0136 | 0.9082 | +1.10 [+0.57, +1.60] | — | 0 |
| B0_seed42 ★ deployed | 3000 | 0.0543 | 0.0316 | 0.0245 | 0.9213 | +3.33 [+2.50, +4.17] | +2.23 [+1.53, +2.97] | 0 |
| B0_seed2026 | 3000 | 0.0530 | 0.0309 | 0.0240 | 0.9213 | +3.20 [+2.37, +4.00] | +2.10 [+1.40, +2.87] | 0 |
| B0_seed7 | 3000 | 0.0523 | 0.0313 | 0.0248 | 0.9212 | +3.13 [+2.30, +3.97] | +2.03 [+1.37, +2.77] | 0 |
| B1_seed42 ★ deployed | 3000 | 0.1317 | 0.0723 | 0.0538 | 0.9527 | +11.07 [+9.83, +12.30] | +9.97 [+8.73, +11.17] | 0 |
| B1_seed2026 | 3000 | 0.1333 | 0.0734 | 0.0548 | 0.9529 | +11.23 [+10.00, +12.47] | +10.13 [+8.93, +11.33] | 0 |
| B1_seed7 | 3000 | 0.1310 | 0.0732 | 0.0552 | 0.9529 | +11.00 [+9.77, +12.27] | +9.90 [+8.73, +11.10] | 0 |
| C0_seed42 ★ deployed | 3000 | 0.1083 | 0.0621 | 0.0478 | 0.9640 | +8.73 [+7.53, +9.93] | +7.63 [+6.53, +8.77] | 0 |
| C0_seed2026 | 3000 | 0.1143 | 0.0635 | 0.0479 | 0.9661 | +9.33 [+8.17, +10.57] | +8.23 [+7.10, +9.47] | 0 |
| C0_seed7 | 3000 | 0.1173 | 0.0698 | 0.0550 | 0.9662 | +9.63 [+8.43, +10.90] | +8.53 [+7.33, +9.80] | 0 |
| AUX_slides_v0 | 3000 | 0.0100 | 0.0044 | 0.0027 | 0.8490 | -1.10 [-1.67, -0.53] | -2.20 [-2.87, -1.53] | 0 |
| AUX_profile_v1 | 3000 | 0.0000 | 0.0000 | 0.0000 | 0.7847 | -2.10 [-2.63, -1.63] | -3.20 [-3.83, -2.57] | 0 |
| B0 three-seed mean ± SD | — | 0.0532 ± 0.0008 | 0.0313 | — | 0.9213 | +3.22 [+2.42, +4.00] | +2.12 [+1.48, +2.84] | — |
| B1 three-seed mean ± SD | — | 0.1320 ± 0.0010 | 0.0730 | — | 0.9528 | +11.10 [+9.89, +12.30] | +10.00 [+8.84, +11.17] | — |
| C0 three-seed mean ± SD | — | 0.1133 ± 0.0037 | 0.0652 | — | 0.9654 | +9.23 [+8.12, +10.34] | +8.13 [+7.07, +9.22] | — |

### 5.2 Key comparisons (user-paired bootstrap, 2,000 resamples, resampled by user)

| Comparison (learned models averaged over three seeds at user level) | ΔHR@8 (pp, 95% CI) | ΔNDCG@8 (pp, 95% CI) |
|---|---|---|
| A1 − A0: gain from weight tuning | +1.10 [+0.57, +1.60] | +0.72 [+0.43, +1.03] |
| B0 − A1: gain from a learned scoring function on the same two features | +2.12 [+1.48, +2.84] | +1.34 [+0.89, +1.81] |
| B1 − B0: gain from adding six features | +7.88 [+6.78, +8.99] | +4.17 [+3.50, +4.86] |
| C0 − B1: content two-tower vs eight-feature GBDT | -1.87 [-3.03, -0.69] | -0.78 [-1.56, -0.02] |
| C0 − B0 | +6.01 [+5.02, +7.01] | +3.39 [+2.74, +4.07] |

- Weight tuning (A1 − A0): ΔHR@8 +1.10 percentage points [+0.57, +1.60], interval above 0: a stable gain.
- Learned scoring function (B0 − A1, same two features): ΔHR@8 +2.12 percentage points [+1.48, +2.84], interval above 0: a stable gain.
- Added features (B1 − B0): ΔHR@8 +7.88 percentage points [+6.78, +8.99], interval above 0: a stable gain.
- Content two-tower vs eight-feature GBDT (C0 − B1): ΔHR@8 -1.87 percentage points [-3.03, -0.69], interval below 0: reliably worse.
- C0 has the highest AUC (0.9640) and B1 the highest HR@8 (0.1317): AUC measures the overall ordering of the target against all weak negatives, HR@8 only looks at the top 8, so the two can name different winners; the pre-registered primary metric is HR@8.

The intervals reflect only user-sampling uncertainty and do not cover proxy-label or static-snapshot bias.

### 5.3 Subgroup results

**Rule scenarios (target eligible)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| R0 | 3600 | 0.0292 | 0.0400 | 0.0675 | 0.1533 | 0.1308 |
| R1 | 1888 | 0.0583 | 0.0757 | 0.1345 | 0.2373 | 0.2161 |
| R2 | 2511 | 0.0382 | 0.0422 | 0.0566 | 0.1430 | 0.1250 |
| R3 | 1255 | 0.0749 | 0.0813 | 0.1155 | 0.2311 | 0.2135 |

**Collection-size group (R0)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| main | 3000 | 0.0210 | 0.0320 | 0.0543 | 0.1317 | 0.1083 |
| sparse | 600 | 0.0700 | 0.0800 | 0.1333 | 0.2617 | 0.2433 |

**Visible history (R0)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| 1-4 | 600 | 0.0700 | 0.0800 | 0.1333 | 0.2617 | 0.2433 |
| 5-9 | 410 | 0.0463 | 0.0659 | 0.0854 | 0.2024 | 0.1610 |
| >=10 | 2590 | 0.0170 | 0.0266 | 0.0494 | 0.1205 | 0.1000 |

**Strict candidate count (R0–R3 pooled, target eligible)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| 9-50 (N<100, too few for a stable comparison) | 3 | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 1.0000 |
| 51-500 (N<100, too few for a stable comparison) | 26 | 0.1923 | 0.2308 | 0.2115 | 0.4808 | 0.4615 |
| >500 | 3600 | 0.0341 | 0.0438 | 0.0807 | 0.1700 | 0.1618 |

**Concentrated / mixed profile (R0)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| concentrated | 1787 | 0.0459 | 0.0599 | 0.0957 | 0.2132 | 0.1701 |
| mixed | 1813 | 0.0127 | 0.0204 | 0.0397 | 0.0943 | 0.0921 |

### 5.4 End-to-end results under the rule scenarios

| Rule | Target eligible share | Empty-result share | Fewer than 8 share | End-to-end A0 | End-to-end A1 | End-to-end B0_seed42 | End-to-end B1_seed42 | End-to-end C0_seed42 |
|---|---|---|---|---|---|---|---|---|
| R0 (no filter) | 100.0% | 0.0% | 0.0% | 0.0292 | 0.0400 | 0.0675 | 0.1533 | 0.1308 |
| R1 (keyword = most common genre in the visible collection) | 52.4% | 0.0% | 0.0% | 0.0306 | 0.0397 | 0.0706 | 0.1244 | 0.1133 |
| R2 (public score ≥ 7.0 and episodes ≤ 26) | 69.8% | 0.0% | 0.0% | 0.0267 | 0.0294 | 0.0394 | 0.0997 | 0.0872 |
| R3 (R1 + R2) | 34.9% | 0.0% | 0.0% | 0.0261 | 0.0283 | 0.0403 | 0.0806 | 0.0744 |

'End-to-end' = the share of all original queries where the strict Top-8 contains the hidden target (a target that is not eligible counts as not met); this is a different quantity from the conditional metrics in 5.3.

Recommendation hygiene (all test rows): A0 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; A1 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; B0_seed42 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; B1_seed42 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; C0_seed42 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%.

### 5.5 Latency and cost

| Model | Scoring only R0 P50/P95 | Scoring only R3 | Service R0 | Service R3 | API R0 | API R3 | Final training time (s) | Model size (KB) |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.024 / 0.030 | 0.005 / 0.005 | 6.9 / 11.3 | 1.7 / 2.5 | 24.0 / 35.4 | 18.5 / 25.9 | 0 | 0 |
| A1 | 0.025 / 0.029 | 0.005 / 0.005 | 6.9 / 11.8 | 1.6 / 2.5 | 24.1 / 36.3 | 18.6 / 25.2 | 0 | 0 |
| B0 | 83.6 / 94.7 | 6.7 / 7.3 | 91.2 / 110.1 | 8.7 / 9.7 | 112.2 / 152.9 | 25.8 / 33.8 | 5.1 | 1699 |
| B1 | 113.9 / 146.3 | 9.5 / 10.4 | 122.9 / 153.2 | 11.4 / 12.8 | 142.6 / 194.0 | 28.1 / 34.3 | 5.4 | 1714 |
| C0 | 0.336 / 0.382 | 0.127 / 0.177 | 0.866 / 0.993 | 0.715 / 0.902 | 17.4 / 23.9 | 17.4 / 24.9 | 26.5 | 482 |

P50 / P95 in milliseconds; measured on one machine (macOS-26.6.2-arm64-arm-64bit, 8 cores, 16 GB memory), 1 CPU thread, no GPU, no result cache, no batching; 200 fixed requests measured after 20 warm-up runs (0 requests without a keyword were skipped under R3). Scoring only = the scorer alone; service = profile + strict rules + features + scoring + Top-8; API = the collection endpoint called in-process (reading the collection and reasons, writing the run log).
Load time (not counted in request latency): catalogue and rule engine 0.64 s; formulas 0.0002 s; B0 0.1161 s; B1 0.0887 s; C0 1.1701 s (including 24905 precomputed anime vectors in 0.0069 s). Peak memory (whole process): after_catalogue 230.7 MB, after_models 595.2 MB, after_inprocess 635.0 MB, after_api 1075.5 MB.
The project has no fixed latency SLO, so only measured values are reported here and they are not a selection threshold. In serving, B0 / B1 use the NumPy reader for the LightGBM text model (native LightGBM deadlocks on an OpenMP conflict when PyTorch is loaded in the same process); the freeze step checked the score gap against native LightGBM (maximum absolute difference 0.0).

## 6. Pre-selection, test confirmation and the default scorer

| Candidate (validation) | HR@8 | seed SD | NDCG@8 | HR@8 per seed |
|---|---|---|---|---|
| A1 | 0.0267 | — | 0.0148 | — |
| B0 | 0.0549 | 0.0017 | 0.0327 | 42: 0.0573, 2026: 0.0540, 7: 0.0533 |
| B1 | 0.1301 | 0.0022 | 0.0773 | 42: 0.1320, 2026: 0.1313, 7: 0.1270 |
| C0 | 0.1051 | 0.0019 | 0.0593 | 42: 0.1053, 2026: 0.1027, 7: 0.1073 |

Pre-selected on the validation set under the pre-registered rule (`selection_protocol.json`, written before any training): **B1**; the deployed artifact is seed 42; the conservative fallback is A1.

| Test confirmation check (deployed artifact B1_seed42) | Threshold | Measured | Result |
|---|---|---|---|
| auc_at_least | 0.65 | 0.9527 | pass |
| hr_gain_vs_A0 | 0.02 | 0.1107 | pass |
| rule_violations_zero | — | 0 | pass |
| ci_lower_vs_A0_above_zero | — | [+9.83, +12.30] (percentage points) | pass |
| ci_lower_vs_A1_above_zero | — | [+8.73, +11.17] (percentage points) | pass |

Conclusion: confirmed. Default scorer = **B1**. Reason: B1 passed every pre-registered confirmation check on test. The test results were not used to retune or to switch to a different model.

Site configuration (`configs/real.yaml`): scorer = `B1`, algorithm2_run = `algorithm2_benchmark/results/real-main-01` — set according to this conclusion.

## 7. Failure cases and limitations

`case_analysis.md` reports 13 real test cases: A0 hits, learned models (B1 and C0) miss 26; A learned model hits, A0 misses 518; All five core models miss 2399; Sparse collections (1–4 visible titles) 600; Mixed preferences (genre entropy above the validation median) 1809; Rule excludes the hidden target (strict mode does not insert it) 5146; Strict empty result (candidate count = 0) 0; Fewer than 8 candidates 0 (occurrences in the test set).

Main limitations:
- Proxy labels: the set of titles rated ≥ 8 is not the same as a collection; unrated titles are weak negatives, so AUC measures how well the target is separated from those weak negatives.
- Static snapshot (2023) and held-out users: what is evaluated is whether other high-rated titles from the same period can be recovered, not a prediction of future behaviour.
- Popularity bias: targets are mostly popular high-rated titles, so features that rely on public score / popularity have a natural advantage; obscure targets rarely reach the top 8.
- Field coverage: theme and demographic depend on a supplement file and are incompletely covered; missing values are treated as NaN / empty sets and never fabricated.
- The LightGBM seed only affects bin-construction sampling (the grid has no subsampling), so differences across the three seeds do not represent the full training randomness.
- C0 could in principle encode new titles with no interactions, but there was no item-disjoint experiment here, so we cannot claim better cold start for new titles.
- Latency was measured on a single laptop and is meant for relative comparison only, not for production.

## 8. System integration verification

| Check (integration_report.json) | Result |
|---|---|
| catalogue_in_app_db | pass |
| service_loaded | pass |
| offline_online_top8 | pass |
| collection_change | pass |
| strict_short_list | pass |
| strict_no_match_keyword | pass |
| strict_no_match | pass |
| explicit_relaxation | pass |
| detail_pages | pass |
| collection_page_renders | pass |
| incompatible_bundle:model_file_changed | pass |
| incompatible_bundle:catalogue_hash_changed | pass |
| missing_bundle_fallback | pass |
| missing_bundle_no_silent_relaxation | pass |

Runtime environment: throwaway copy of the activated real database (evaluation/sandbox.py); the business database was not modified. Overall status: **passed**.

Integrated: the collection endpoint reads the current collection H and calls exactly the same catalogue, profile, rule engine, features and scorer as the offline path; the scorer registry offers A0 / A1 / B0 / B1 / C0 (`recommendation.collection.scorer`, or the `scorer` field in the request body for experiments / operations); a missing or incompatible model bundle falls back to the formulas and records the status without relaxing the user's filters; when a strict query returns nothing it offers relaxation only, applies it after the user confirms, and the response carries `relaxed`, `relaxed_rules` and `strict_result_count`, with a readable relaxation note on the page.
Still open: the application database holds no real user collections and there is no online experiment (A/B or click feedback), so the actual collection recommendation quality on the site is not yet validated; every effectiveness number in this report comes from the offline proxy evaluation.

## 9. Reproduction

See `README_reproduce.md` in the same directory (commands, data locations and output files). The main records: `config.json`, `data_manifest.json`, `split_manifest.json`, `query_manifest.parquet`, `sampled_pairs.parquet`, `selection_protocol.json`, `trial_results.csv`, `tuning.json`, `training_curves.json`, `feature_importance.csv`, `models_manifest.json`, `models/`, `freeze_manifest.json`, `feature_schema.json`, `predictions.parquet`, `metrics.json`, `summary.csv`, `subgroup_metrics.csv`, `paired_comparisons.csv`, `decision.json`, `latency.csv`, `benchmark.json`, `integration_report.json`, `case_analysis.md`, `run_metadata.json`.

