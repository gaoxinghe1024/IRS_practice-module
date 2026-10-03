# Delivery status

State of the rebuild on branch `anilogic-rebuild`, 2026-10-01. Real dataset version `real-89bfdea00b` (Kaggle MAL 2023,
all 268,600 offline users, no sampling); synthetic version `synthetic-7a7d5d6a74`. Status terms follow the brief:
implemented → tested_synthetic → tested_real → real_data_evaluated, or limited_by_data / limited_by_credentials.
Nothing here is described as production ready: this is a course system evaluated offline on one machine.

## 1. Capabilities

| Capability | Status | Evidence | What is missing |
|---|---|---|---|
| Accounts, sessions, isolation | implemented, tested_synthetic, tested_real | `test_auth_security.py`, `test_isolation.py`, journey 12, real sandbox test | - |
| Catalogue search / detail / sources / missing fields | implemented, tested_synthetic, tested_real | journeys 1-2, `test_real_data.py`, system eval (search p50 3.0 ms) | - |
| Ratings, collections, progress, exclusions | implemented, tested_synthetic, tested_real | `test_user_data_and_recs.py` (incl. restart persistence), journeys 3, 5, 11 | - |
| Algorithm 1 — similar titles (A default; B, C, D) | implemented, tested_synthetic, tested_real, real_data_evaluated (franchise sanity + weak labels) | `similar` runs; A Hits@5 0.652 on 23 franchise queries | human similarity labels (`data/evaluation/similarity_labels.jsonl`); B is limited_by_data on real data (2 weak-label queries) |
| Algorithm 2 — collection recommendations (A0 / A1 formulas, B0 / B1 LightGBM, C0 two-tower; B1 default in real mode) | implemented, tested_synthetic, tested_real, real_data_evaluated on a proxy label (rating >= 8 sets, not collections) | `algorithm2_benchmark/results/real-main-01` (24,000 users, frozen protocol, test confirmation); `integration_report.json`; `algorithm2_benchmark/tests/`, `test_collection_algorithm2.py`; journey 4 | real collection events for a non-proxy evaluation; online feedback |
| Algorithm 3 — home: cold start, ICF Top-20 recall, GA Top-10 | implemented, tested_synthetic, tested_real, real_data_evaluated | `homepage`, `explicit`, `ga_ablation` runs; journeys 5-6 | - (see §2 for what the numbers say) |
| Algorithm 4 — chat-to-find | implemented, tested_synthetic, tested_real, real_data_evaluated (auto query sets) | `chat` and `system` runs; journeys 7, 8, 10 | human query set (`data/evaluation/rag_queries.jsonl`); **LLM answers limited_by_credentials** (local mode works without them) |
| Viewing plan | implemented, tested_synthetic, tested_real, real_data_evaluated | `planner` run: 0 infeasible among 2,400 plans (greedy + 3 GA seeds, 120 users x 5 requests); journey 9 | - |
| Explanations, cold start, feedback, history, status | implemented, tested_synthetic, tested_real | tests listed in `docs/feature-traceability.md` | - |
| Relation graph | implemented, tested_synthetic, tested_real; status `degraded` | 177 relations from 33 titles (partial Jikan layer) | complete relation data (`data/enriched/relations.jsonl` or a Jikan run when MyAnimeList is reachable) |
| Scene search | implemented, tested_synthetic; limited_by_data | synthetic scene queries; real refusals (0 fabricated episodes in 30 real queries) | `data/enriched/scenes.jsonl` with sources |
| Screenshot identification | implemented, tested_synthetic (pipeline only); limited_by_data | fixture frames; real run `not_run` | `data/enriched/scene_images/manifest.jsonl` + frames; no accuracy is claimed |
| Legacy account migration | implemented, tested_synthetic, dry run on real | `test_legacy_migration.py`; audit §4 | decision on the 11 seed-pattern ratings of the one legacy account |
| Docker packaging | implemented, **not verified** | `Dockerfile`, `compose.yaml` | Docker is not installed on the delivery machine |

## 2. What the real-data evaluation shows

All numbers come from `docs/evaluation-results-real.md`, which is generated from the run directories. The brackets
are 95 % bootstrap intervals.

**Home recommendations (ranking protocol, 1,500 test users, K = 10).**

| Engine | Hits@10 no-rerank / greedy | Hits@10 GA | Candidate recall@20 | Catalogue coverage |
|---|---|---|---|---|
| ICF (served) | 0.062 | 0.053 | 0.087 | 0.18 |
| MF | 0.041 | 0.030 | 0.068 | 0.13 |
| NCF | 0.037 | 0.031 | 0.065 | 0.01 |
| Popularity | 0.137 | 0.132 | 0.200 | 0.006 |

* **Popularity is the strongest accuracy baseline** on this protocol, about twice ICF. The held-out positive is a
  title the user rated >= 8, and those are mostly popular titles. ICF is the best *personalised* engine and covers
  about 30 times more of the catalogue. The home page therefore shows a clearly labelled, separate "Explore" list of
  popular titles next to the personalised list.
* **GA vs greedy on the same pool:** Hits@10 −0.009 [−0.017, −0.003], diversity (ILD) +0.056, novelty +0.036, fresh
  share +0.072. The GA reaches the exact optimum of its objective in 99.8 % of small pools, and runs in 26 ms p50.
  The ablation shows that the **novelty term causes most of the accuracy loss**: without it the GA matches
  greedy's Hits@10 (0.040 vs 0.038 on validation) and keeps most of the diversity gain. The weights are the
  proposal's (0.4 / 0.3 / 0.2 / 0.1) and were not changed. Lowering the novelty weight is an open product decision
  that the data supports.
* **ICF tuning:** damping 20 and user-mean prior weight 5, chosen on validation positives of a disjoint user group
  (`configs/real.yaml` names the run). Without the prior, a new user who gives every title the same rating gets
  no personalisation signal at all.
* **Short histories (warm-gate analysis, run `20261001-034040-real-homepage-99cd`):** with 1-30 ratings every
  personalised ranking stays far below popularity (0.12-0.14). The tag-only content profile was close to 0
  (0.009 validation Hits@10 over 1-5 ratings), so the cold-start content score now blends in quality-controlled
  popularity with a validation-tuned weight g = 0.9 (0.041). On test users it reaches 0.034-0.056, against 0.016-0.034 for
  ICF. **Finding:** on this accuracy metric the partial-mode blend (below 8 ratings) and pure ICF (from 8 ratings)
  are no better than the popularity-weighted content ranking. They add personalisation and coverage, which Hits@10
  on popularity-biased positives does not reward. Raising the warm gate, or adding a popularity prior to warm-mode
  relevance, are data-supported options left as product decisions.

**Collection recommendations (algorithm 2, run `real-main-01`, 3,000 test users, full catalogue of 14,182 titles,
user-macro HR@8 on rule R0).** The label is a proxy: the user's titles rated >= 8 act as the collection, one is
hidden. A0 (the slides' formula) 0.021, A1 (validation weight 0.05) 0.032, B0 (LightGBM on the same two features)
0.053, **B1 (LightGBM, eight features) 0.132**, C0 (two-tower) 0.113 (three-seed means for B and C). B1 was
pre-selected on validation and passed every pre-registered check on test: +11.1 points over A0 [+9.8, +12.3] and
+10.0 over A1 [+8.7, +11.2], AUC 0.953, no rule violations. It is the real-mode default. The extra features carry
most of the gain (B1 − B0 +7.9 [+6.8, +9.0]); the two-tower ranks below B1 (−1.9 [−3.0, −0.7]). Serving cost on
this machine: B1 120 ms p50 for a full-catalogue request in the service (formulas 7 ms, C0 1 ms). The smoke run
(150 test users) chose differently (A1 kept), which is why only the main run decides. Details:
`algorithm2_benchmark/results/real-main-01/final_report.md`.

**Rating prediction (explicit protocol, 239,937 test pairs):** MF MAE 0.872 / RMSE 1.168; NCF 0.920 / 1.241; ICF
(exact formula) 0.966 / 1.347; user mean 1.100 / 1.435. The damping that helps ICF ranking makes its rating
predictions worse (MAE 1.034), so the two protocols are reported separately. NCF did not overfit: it is
regularised and early-stopped, and it trails MF.

**Chat (local mode, auto-generated query sets, not human relevance judgements):** title lookup (exact / alias /
one-typo) Hits@5 0.968 over 312 queries, up from 0.917 after fixing sequels that differ only in punctuation and
titles containing "," or "!". The remaining misses are short one-letter typos, an ambiguous typo that correctly asks
which title is meant, and one title shared by two series. Constraint requests 38 / 38 answered and 100 % compliant;
sequel questions Hits@5 1.0 (17); unanswerable scene-episode questions 30 / 30 refused with no fabricated episode
number.

**Viewing plans (120 users x 5 request types):** 0 infeasible plans; GA fitness 0.82-0.84 vs greedy 0.61-0.64 on the
same objective; exact-optimal in 95.3 % of 12-unit pools (mean gap 0.0007).

**System (single machine, in-process):** p50 / p95 ms — search 3.0 / 4.7, detail 4.1 / 5.4, similar 6.2 / 7.2 (cold),
home 36.9 / 40.3 (cold) and 3.3 / 4.1 (cached), chat 6.0 / 38.8, plan preview 187.7 / 196.5. Five concurrent readers:
p95 34.5 ms, 0 errors; 90 concurrent rating writes, 0 failures. All five injected LLM failures (timeout, 429,
invalid JSON, empty, refusal) fall back to local answers; a failing dense encoder falls back to lexical search.

## 3. Default algorithms and why

| Surface | Default | Why |
|---|---|---|
| Similar titles | A: weighted attributes | Always available and explainable; the alternatives cannot yet be judged on real data (no human labels; the replacement rule makes no decision with fewer than 20 validation queries) |
| Collection | B1 (LightGBM, eight features; real mode) / A0 (synthetic mode) | Pre-selected on validation and confirmed on test by the pre-registered rule (beats A0 and the tuned formula A1 with paired 95 % intervals above zero); falls back to the formula, with the filters still strict, if the frozen run cannot be verified. The label is a proxy (high-rating sets), not real collections |
| Home recall | ICF (damping 20, prior weight 5) | Best personalised engine on the real ranking protocol, and works for new users without retraining; MF folds in users but ranks lower; NCF cannot serve new users (ID-only) |
| Home list | GA (proposal weights) | Improves the stated list objective at a small, measured accuracy cost; greedy is one click away on `/dev` for comparison |
| Chat | deterministic parser + hybrid retrieval + local grounded answers | Works without credentials; an LLM adds wording only after validation |
| Plans | GA with exact oracle check on small pools | Near-optimal and always feasible |

## 4. Not done / blocked

* **Human evaluation** (similarity labels, chat relevance, task-success sessions): protocol ready in
  `data/evaluation/human-evaluation-protocol.md`; not run. Status: human_evaluation_pending.
* **LLM answers:** the adapters and the answer validation are implemented and tested against stubbed responses; a real
  run needs `ANILOGIC_LLM__PROVIDER` / `ANILOGIC_LLM__MODEL` / `ANILOGIC_LLM_API_KEY`. **Plain LLM vs RAG:** the Plain LLM baseline (same model, no retrieval) and its comparison harness are **not implemented yet**; the chat experiment only records the status. Both need a model key and a budget approval.
  No paid calls were made. A Plain LLM vs RAG vs template benchmark of the AnimeFinder v1.4.1 chat pipeline is in `algorithm4_benchmark/` (its model answers came through a CLI channel, not vendor API keys); the same comparison for `app/chat` is still open.
* **Jikan enrichment:** MyAnimeList returned HTTP 504 for most requests on 2026-09-30; `make enrich-jikan` resumes.
* **Scenes, frames, real collections:** no data in the snapshot; contracts, validators and pipelines are ready. The
  collection recommender is therefore selected on a proxy label; its effect on real collections is not measured.
* **Docker:** not run on the delivery machine.

## 5. Where the evidence is

`docs/test-report.md` (tests), `docs/evaluation-results-real.md` and `docs/evaluation-results-synthetic.md`
(generated summaries), `evaluation/runs/<run_id>/` (config, manifest, metrics, per-case rows, failures, report),
`algorithm2_benchmark/` (algorithm 2: pipeline, runs with protocol, models, predictions and reports, its own tests),
`docs/screenshots/` (real-mode pages) and `docs/screenshots/e2e/` (browser journeys), `var/<mode>/pipeline_runs/`
(pipeline records; local).
