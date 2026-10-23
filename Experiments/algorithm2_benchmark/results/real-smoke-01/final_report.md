# Algorithm 2 (collection recommendation): effectiveness comparison, model selection and system integration report — smoke-level pipeline check (not a formal model-selection conclusion)

Run `real-smoke-01` · level **smoke** · generated at 2026-10-02T08:46:05Z. Every number in this report comes from a record file in this run directory (see section 9); anything that did not run is written as NOT_RUN.

> Note: this is a smoke-level run with few users; it only checks the pipeline and is not a formal model-selection conclusion.

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
- This level uses 1,200 users (nested in the full split: True). Users shared between partitions: {'train&val': 0, 'train&test': 0, 'val&test': 0}.
- Queries: one hidden target per user for validation / test, up to 3 targets per user for training; 2,837 queries in total ({'main': {'test': 150, 'train': 2100, 'val': 150}, 'sparse': {'test': 30, 'train': 377, 'val': 30}}). The hidden target never enters the profile input H.
- Training samples: 1 positive + 4 weak negatives per training query (deterministic sampling inside the same candidate set, `negative_seed = 42`, excluding all known positives of that user), 13,285 pairs in total (12,385 training pairs; the 900 pairs of validation users are used only for the early-stopping logloss); the negative shortfall occurred 0 times. B0 / B1 / C0 share the same sample set.
- Rule scenarios: R0 has no filter; R1 takes the most common genre in H as the keyword (ties broken by name), and the keyword is matched as an AND substring against genre or title; R2 requires public score ≥ 7.0 and episodes ≤ 26 (an unknown episode count does not satisfy it); R3 = R1 + R2. All of them are strict: a target that does not satisfy the rule is not inserted back, is recorded as target_not_eligible, and end-to-end results are reported separately.
- Candidates: the **full catalogue** passing the rule is ranked (no candidate sampling), excluding H itself. Ties are broken by ascending MAL id; AUC gives 0.5 to ties and is unaffected by the id ordering; AUC is null when there are fewer than 2 candidates.
- Primary metric (pre-registered): user-macro HR@8 on R0, main group, more than 8 candidates, target eligible; NDCG@8, MRR@8 and AUC are reported in the same table.

## 4. Implementation, parameters and budget of the five experiment groups

- **A0** = rating_norm + 0.15 × hits (hits = the number of candidate genre/theme tags that fall inside the top-10 namespace tags of H).
- **A1**: hits weights [0.05, 0.1, 0.15, 0.2, 0.3] compared on the validation set, selected **0.05** (validation HR@8 0.0333); the selected value is the minimum of the pre-registered grid, which suggests the data may prefer a weight outside the grid, but the grid was not extended after the results were seen.
- **B0**: native LightGBM `lgb.train` (objective = binary, ranked by raw score), features rating_norm, hits; 10 pre-generated deterministic trials (grid {'num_leaves': [15, 31], 'max_depth': [5, 8], 'learning_rate': [0.03, 0.05], 'min_child_samples': [50, 100], 'reg_lambda': [1.0, 5.0]}), capped at 500 trees with early stopping after 50 rounds (logloss on validation), trial 5 selected by validation HR@8 over the full candidate set: {'learning_rate': 0.05, 'max_depth': 5, 'min_child_samples': 50, 'num_leaves': 31, 'reg_lambda': 1.0} (validation HR@8 0.0400).
- **B1**: native LightGBM `lgb.train` (objective = binary, ranked by raw score), features rating_norm, hits, genre_affinity, theme_affinity, demographic_affinity, studio_affinity, length_affinity, max_tag_similarity; 10 pre-generated deterministic trials (grid {'num_leaves': [15, 31], 'max_depth': [5, 8], 'learning_rate': [0.03, 0.05], 'min_child_samples': [50, 100], 'reg_lambda': [1.0, 5.0]}), capped at 500 trees with early stopping after 50 rounds (logloss on validation), trial 8 selected by validation HR@8 over the full candidate set: {'learning_rate': 0.05, 'max_depth': 8, 'min_child_samples': 100, 'num_leaves': 15, 'reg_lambda': 1.0} (validation HR@8 0.1000).
- **C0**: PyTorch content two-tower, [128, 64] hidden units per side; inputs are multi-hot vectors over genre / theme / demographic / studio (studios with at least 5 appearances, the rest mapped to UNK) plus numeric features and a missing-value mask (anime tower 378 dims, collection tower 380 dims), dot-product logits + BCEWithLogitsLoss, no in-batch negatives; 10 trials, trial 1 selected: {'dropout': 0.2, 'learning_rate': 0.0003, 'out_dim': 64, 'weight_decay': 0.0001} (validation HR@8 0.1000), at most 30 epochs, patience 3, on CPU.
- The final configuration is trained once per training seed [42, 2026, 7] (9 models in total, see models_manifest.json). Note: the pre-registered LightGBM grid has no bagging / feature_fraction subsampling, so the seed only acts through bin-construction sampling; the 12,385 training pairs do not exceed LightGBM's histogram bin-construction sample cap of 200,000, so binning uses every sample and the trees of the three B0 and B1 seeds are identical (a standard deviation of 0 is a real property of this configuration, not three independent random draws).
- Tuning cost (B0 10 trials, B1 10 trials, C0 10 trials, total): B0 fit 8 s + full-candidate validation scoring 20 s; B1 fit 10 s + full-candidate validation scoring 29 s; C0 fit 11 s (validation scoring not timed separately). Validation scoring over the full catalogue is the main cost.
- Comparison logic: B0 is the two-feature control; B1 and C0 use metadata from the same source but different representations, so C0 vs B1 is not 'the same input with only the model swapped'.
- B1 feature importance (gain share, deploy seed): rating_norm 61.0%, studio_affinity 11.8%, hits 6.9%, length_affinity 6.5%, max_tag_similarity 4.3%, theme_affinity 3.6%, demographic_affinity 3.0%, genre_affinity 2.9%. Importance is not causal contribution; a feature's effect is judged by the B1 − B0 comparison.

## 5. Test results

Test set: 180 users and 180 queries. The main table covers the pre-registered primary set (R0, main group, >8 candidates, target eligible; user-macro).

### 5.1 Main table

| Model | Users | HR@8 | NDCG@8 | MRR@8 | AUC | ΔHR@8 vs A0 (pp, 95% CI) | ΔHR@8 vs A1 (pp, 95% CI) | Rule violations |
|---|---|---|---|---|---|---|---|---|
| A0 | 150 | 0.0200 | 0.0142 | 0.0122 | 0.8819 | — | -1.33 [-4.00, +1.33] | 0 |
| A1 | 150 | 0.0333 | 0.0181 | 0.0130 | 0.9163 | +1.33 [-1.33, +4.00] | — | 0 |
| B0_seed42 ★ deployed | 150 | 0.0467 | 0.0265 | 0.0205 | 0.9237 | +2.67 [-1.33, +7.33] | +1.33 [-2.00, +5.33] | 0 |
| B0_seed2026 | 150 | 0.0467 | 0.0265 | 0.0205 | 0.9237 | +2.67 [-1.33, +7.33] | +1.33 [-2.00, +5.33] | 0 |
| B0_seed7 | 150 | 0.0467 | 0.0265 | 0.0205 | 0.9237 | +2.67 [-1.33, +7.33] | +1.33 [-2.00, +5.33] | 0 |
| B1_seed42 ★ deployed | 150 | 0.1000 | 0.0558 | 0.0423 | 0.9570 | +8.00 [+2.67, +13.33] | +6.67 [+2.00, +11.33] | 0 |
| B1_seed2026 | 150 | 0.1000 | 0.0558 | 0.0423 | 0.9570 | +8.00 [+2.67, +13.33] | +6.67 [+2.00, +11.33] | 0 |
| B1_seed7 | 150 | 0.1000 | 0.0558 | 0.0423 | 0.9570 | +8.00 [+2.67, +13.33] | +6.67 [+2.00, +11.33] | 0 |
| C0_seed42 ★ deployed | 150 | 0.0667 | 0.0290 | 0.0172 | 0.9317 | +4.67 [+0.67, +8.67] | +3.33 [-1.33, +8.67] | 0 |
| C0_seed2026 | 150 | 0.0533 | 0.0373 | 0.0321 | 0.9289 | +3.33 [-0.67, +7.33] | +2.00 [-2.00, +6.67] | 0 |
| C0_seed7 | 150 | 0.0733 | 0.0344 | 0.0222 | 0.9291 | +5.33 [+0.67, +10.00] | +4.00 [-0.67, +8.67] | 0 |
| AUX_slides_v0 | 150 | 0.0067 | 0.0022 | 0.0010 | 0.8466 | -1.33 [-4.00, +1.33] | -2.67 [-6.00, +0.00] | 0 |
| AUX_profile_v1 | 150 | 0.0000 | 0.0000 | 0.0000 | 0.8238 | -2.00 [-4.67, +0.00] | -3.33 [-6.67, -0.67] | 0 |
| B0 three-seed mean ± SD | — | 0.0467 ± 0.0000 | 0.0265 | — | 0.9237 | +2.67 [-1.33, +7.33] | +1.33 [-2.00, +5.33] | — |
| B1 three-seed mean ± SD | — | 0.1000 ± 0.0000 | 0.0558 | — | 0.9570 | +8.00 [+2.67, +13.33] | +6.67 [+2.00, +11.33] | — |
| C0 three-seed mean ± SD | — | 0.0644 ± 0.0083 | 0.0336 | — | 0.9299 | +4.44 [+0.88, +8.22] | +3.11 [-1.11, +7.56] | — |

### 5.2 Key comparisons (user-paired bootstrap, 2,000 resamples, resampled by user)

| Comparison (learned models averaged over three seeds at user level) | ΔHR@8 (pp, 95% CI) | ΔNDCG@8 (pp, 95% CI) |
|---|---|---|
| A1 − A0: gain from weight tuning | +1.33 [-1.33, +4.00] | +0.39 [-0.91, +1.83] |
| B0 − A1: gain from a learned scoring function on the same two features | +1.33 [-2.00, +5.33] | +0.85 [-1.34, +3.17] |
| B1 − B0: gain from adding six features | +5.33 [+0.67, +10.67] | +2.93 [+0.38, +5.93] |
| C0 − B1: content two-tower vs eight-feature GBDT | -3.56 [-9.78, +2.01] | -2.23 [-6.12, +1.26] |
| C0 − B0 | +1.78 [-1.33, +4.89] | +0.70 [-1.71, +2.98] |

- Weight tuning (A1 − A0): ΔHR@8 +1.33 percentage points [-1.33, +4.00], interval includes 0: not enough evidence to claim a difference.
- Learned scoring function (B0 − A1, same two features): ΔHR@8 +1.33 percentage points [-2.00, +5.33], interval includes 0: not enough evidence to claim a difference.
- Added features (B1 − B0): ΔHR@8 +5.33 percentage points [+0.67, +10.67], interval above 0: a stable gain.
- Content two-tower vs eight-feature GBDT (C0 − B1): ΔHR@8 -3.56 percentage points [-9.78, +2.01], interval includes 0: not enough evidence to claim a difference.

The intervals reflect only user-sampling uncertainty and do not cover proxy-label or static-snapshot bias.

### 5.3 Subgroup results

**Rule scenarios (target eligible)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| R0 | 180 | 0.0167 | 0.0389 | 0.0500 | 0.1056 | 0.0722 |
| R1 (N<100, too few for a stable comparison) | 99 | 0.0404 | 0.0707 | 0.0909 | 0.1313 | 0.1010 |
| R2 | 126 | 0.0238 | 0.0317 | 0.0476 | 0.0873 | 0.0635 |
| R3 (N<100, too few for a stable comparison) | 64 | 0.0469 | 0.0469 | 0.0625 | 0.1094 | 0.1094 |

**Collection-size group (R0)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| main | 150 | 0.0200 | 0.0333 | 0.0467 | 0.1000 | 0.0667 |
| sparse (N<100, too few for a stable comparison) | 30 | 0.0000 | 0.0667 | 0.0667 | 0.1333 | 0.1000 |

**Visible history (R0)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| 1-4 (N<100, too few for a stable comparison) | 30 | 0.0000 | 0.0667 | 0.0667 | 0.1333 | 0.1000 |
| 5-9 (N<100, too few for a stable comparison) | 21 | 0.0000 | 0.0000 | 0.0952 | 0.1905 | 0.0476 |
| >=10 | 129 | 0.0233 | 0.0388 | 0.0388 | 0.0853 | 0.0698 |

**Strict candidate count (R0–R3 pooled, target eligible)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| 51-500 (N<100, too few for a stable comparison) | 1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| >500 | 180 | 0.0222 | 0.0444 | 0.0694 | 0.1111 | 0.0931 |

**Concentrated / mixed profile (R0)** (HR@8, user-level mean)

| Group | Users | A0 | A1 | B0_seed42 | B1_seed42 | C0_seed42 |
|---|---|---|---|---|---|---|
| concentrated (N<100, too few for a stable comparison) | 93 | 0.0108 | 0.0430 | 0.0430 | 0.1398 | 0.0430 |
| mixed (N<100, too few for a stable comparison) | 87 | 0.0230 | 0.0345 | 0.0575 | 0.0690 | 0.1034 |

### 5.4 End-to-end results under the rule scenarios

| Rule | Target eligible share | Empty-result share | Fewer than 8 share | End-to-end A0 | End-to-end A1 | End-to-end B0_seed42 | End-to-end B1_seed42 | End-to-end C0_seed42 |
|---|---|---|---|---|---|---|---|---|
| R0 (no filter) | 100.0% | 0.0% | 0.0% | 0.0167 | 0.0389 | 0.0500 | 0.1056 | 0.0722 |
| R1 (keyword = most common genre in the visible collection) | 55.0% | 0.0% | 0.0% | 0.0222 | 0.0389 | 0.0500 | 0.0722 | 0.0556 |
| R2 (public score ≥ 7.0 and episodes ≤ 26) | 70.0% | 0.0% | 0.0% | 0.0167 | 0.0222 | 0.0333 | 0.0611 | 0.0444 |
| R3 (R1 + R2) | 35.6% | 0.0% | 0.0% | 0.0167 | 0.0167 | 0.0222 | 0.0389 | 0.0389 |

'End-to-end' = the share of all original queries where the strict Top-8 contains the hidden target (a target that is not eligible counts as not met); this is a different quantity from the conditional metrics in 5.3.

Recommendation hygiene (all test rows): A0 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; A1 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; B0_seed42 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; B1_seed42 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%; C0_seed42 rule-violation rate 0% / duplicate rate 0% / already-saved duplicate rate 0%.

### 5.5 Latency and cost

| Model | Scoring only R0 P50/P95 | Scoring only R3 | Service R0 | Service R3 | API R0 | API R3 | Final training time (s) | Model size (KB) |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.023 / 0.028 | 0.004 / 0.005 | 6.5 / 10.5 | 1.6 / 2.6 | 20.3 / 31.7 | 16.4 / 25.7 | 0 | 0 |
| A1 | 0.023 / 0.027 | 0.004 / 0.005 | 6.5 / 10.2 | 1.6 / 2.3 | 20.4 / 29.5 | 15.8 / 21.8 | 0 | 0 |
| B0 | 40.9 / 43.2 | 3.6 / 4.3 | 47.7 / 52.1 | 5.5 / 6.5 | 61.7 / 74.5 | 19.5 / 27.2 | 0.8 | 483 |
| B1 | 58.2 / 66.6 | 5.2 / 5.6 | 65.1 / 74.5 | 7.0 / 7.8 | 80.5 / 111.3 | 21.2 / 40.3 | 1.0 | 436 |
| C0 | 0.328 / 0.345 | 0.140 / 0.273 | 0.849 / 0.930 | 0.749 / 1.3 | 13.9 / 19.5 | 13.6 / 18.5 | 1.6 | 482 |

P50 / P95 in milliseconds; measured on one machine (macOS-26.6.2-arm64-arm-64bit, 8 cores, 16 GB memory), 1 CPU thread, no GPU, no result cache, no batching; 150 fixed requests measured after 20 warm-up runs (0 requests without a keyword were skipped under R3). Scoring only = the scorer alone; service = profile + strict rules + features + scoring + Top-8; API = the collection endpoint called in-process (reading the collection and reasons, writing the run log).
Load time (not counted in request latency): catalogue and rule engine 0.56 s; formulas 0.0001 s; B0 0.0339 s; B1 0.0178 s; C0 0.8121 s (including 24905 precomputed anime vectors in 0.0057 s). Peak memory (whole process): after_catalogue 231.0 MB, after_models 531.4 MB, after_inprocess 613.5 MB, after_api 822.5 MB.
The project has no fixed latency SLO, so only measured values are reported here and they are not a selection threshold. In serving, B0 / B1 use the NumPy reader for the LightGBM text model (native LightGBM deadlocks on an OpenMP conflict when PyTorch is loaded in the same process); the freeze step checked the score gap against native LightGBM (maximum absolute difference 0.0).

## 6. Pre-selection, test confirmation and the default scorer

| Candidate (validation) | HR@8 | seed SD | NDCG@8 | HR@8 per seed |
|---|---|---|---|---|
| A1 | 0.0333 | — | 0.0209 | — |
| B0 | 0.0400 | 0.0000 | 0.0151 | 42: 0.0400, 2026: 0.0400, 7: 0.0400 |
| B1 | 0.1000 | 0.0000 | 0.0417 | 42: 0.1000, 2026: 0.1000, 7: 0.1000 |
| C0 | 0.1089 | 0.0083 | 0.0570 | 42: 0.1000, 2026: 0.1067, 7: 0.1200 |

Pre-selected on the validation set under the pre-registered rule (`selection_protocol.json`, written before any training): **C0**; the deployed artifact is seed 42; the conservative fallback is A1.

| Test confirmation check (deployed artifact C0_seed42) | Threshold | Measured | Result |
|---|---|---|---|
| auc_at_least | 0.65 | 0.9317 | pass |
| hr_gain_vs_A0 | 0.02 | 0.0467 | pass |
| rule_violations_zero | — | 0 | pass |
| ci_lower_vs_A0_above_zero | — | [+0.67, +8.67] (percentage points) | pass |
| ci_lower_vs_A1_above_zero | — | [-1.33, +8.67] (percentage points) | fail |

Conclusion: not confirmed. Default scorer = **A1**. Reason: C0 did not pass ci_lower_vs_A1_above_zero; the pre-registered fallback A1 is deployed. The test results were not used to retune or to switch to a different model.

Site configuration (`configs/real.yaml`): scorer = `B1`, algorithm2_run = `algorithm2_benchmark/results/real-main-01` — not yet pointing at this run (this run is not the frozen result the site currently serves).

## 7. Failure cases and limitations

`case_analysis.md` reports 13 real test cases: A0 hits, learned models (B1 and C0) miss 2; A learned model hits, A0 misses 23; All five core models miss 124; Sparse collections (1–4 visible titles) 30; Mixed preferences (genre entropy above the validation median) 87; Rule excludes the hidden target (strict mode does not insert it) 251; Strict empty result (candidate count = 0) 0; Fewer than 8 candidates 0 (occurrences in the test set).

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

