# Delivery status

State on branch `yyc-data-package`, 2026-10-04: the AniLogic rebuild plus the team data package. Real dataset
version `real-034ebd6a74` (Kaggle MAL 2023, all 268,600 offline users, no sampling, with the team package's
supplementary layers); synthetic version `synthetic-7a7d5d6a74`. Status terms follow the brief: implemented →
tested_synthetic → tested_real → real_data_evaluated, or limited_by_data / limited_by_credentials. Nothing here is
described as production ready: this is a course system evaluated offline on one machine.

What the team package changed (details in `docs/data-contracts.md` §4.1): themes for 14,242 titles and
demographics for 10,395 (before: 33 titles), 19,478 confirmed title-to-title relations from 10,883 titles (before:
177 from 33), character lists for 11,494 titles, 843 synopses where the snapshot had none (missing synopses 18.2 %
→ 14.8 %), 300 sourced scenes for 100 titles (before: none) and 100 evaluation queries. Scores, genres, studios,
episodes and durations are still exactly the 2023 snapshot's, so the frozen algorithm-2 model still verifies
(catalogue hash `24c83192...` unchanged). Recommendable titles: 12,429 → 13,798 (a scored title with a theme but no
genre now qualifies).

## 1. Capabilities

| Capability | Status | Evidence | What is missing |
|---|---|---|---|
| Accounts, sessions, isolation | implemented, tested_synthetic, tested_real | `test_auth_security.py`, `test_isolation.py`, journey 12, real sandbox test | - |
| Catalogue search / detail / sources / missing fields, start-year filter | implemented, tested_synthetic, tested_real | journeys 1-2 (incl. the year filter), `test_real_data.py`, system eval (search p50 3.3 ms) | - |
| Ratings, collections, progress, exclusions | implemented, tested_synthetic, tested_real | `test_user_data_and_recs.py` (incl. restart persistence), journeys 3, 5, 11 | - |
| Algorithm 1 — similar titles (A, B, C, D; **B served in real mode**) | implemented, tested_synthetic, tested_real, real_data_evaluated (389 weak-label queries with a frozen split; franchise sanity on 4,507 titles) | `similar` run `20261002-215313-real-similar-7127`; replacement rule on validation | human similarity labels (`data/evaluation/similarity_labels.jsonl`) |
| Algorithm 2 — collection recommendations (A0 / A1 formulas, B0 / B1 LightGBM, C0 two-tower; B1 default in real mode) | implemented, tested_synthetic, tested_real, real_data_evaluated on a proxy label (rating >= 8 sets, not collections) | `algorithm2_benchmark/results/real-main-01` (24,000 users, frozen protocol, test confirmation); bundle verified on `real-034ebd6a74`; `algorithm2_benchmark/tests/`, `test_collection_algorithm2.py`; journey 4 | real collection events for a non-proxy evaluation; online feedback |
| Algorithm 3 — home: cold start, ICF Top-20 recall, GA Top-10 | implemented, tested_synthetic, tested_real, real_data_evaluated | `homepage`, `explicit`, `ga_ablation` runs; journeys 5-6 | - (see §2 for what the numbers say) |
| Algorithm 4 — chat-to-find | implemented, tested_synthetic, tested_real, real_data_evaluated (team package query set + auto sets; not human relevance judgements) | `chat` run `20261004-170011-real-chat-4e18`, `system` run; journeys 7, 8, 10 | human relevance judgements; **LLM answers implemented but not evaluated** (no approved budget); Plain LLM vs RAG not implemented |
| Viewing plan | implemented, tested_synthetic, tested_real, real_data_evaluated | `planner` run: 0 infeasible among 2,400 plans (greedy + 3 GA seeds, 120 users x 5 requests); journey 9 | - |
| Explanations, cold start, feedback, history, status | implemented, tested_synthetic, tested_real | tests listed in `docs/feature-traceability.md` | - |
| Relation graph | implemented, tested_synthetic, tested_real, real_data_evaluated (60 sequel questions); status `degraded` | 19,478 relations from 10,883 of 23,363 served titles (team package + Jikan); prerequisites in release order | relations for the other titles (the package has none for them: their empty relation field comes from the 2023 snapshot, so "unknown", not "none") |
| Scene search | implemented, tested_synthetic, tested_real, real_data_evaluated; status `degraded` | 300 scenes for 100 titles; team scene queries, out-of-corpus and mismatch question sets, scene-corpus ablation | a human spot check of the scenes (an AI agent checked them against cited sources); scenes for other titles |
| Screenshot identification | implemented, tested_synthetic (pipeline only); limited_by_data | fixture frames; real run `not_run` | `data/enriched/scene_images/manifest.jsonl` + frames; no accuracy is claimed |
| Legacy account migration | implemented, tested_synthetic, dry run on real | `test_legacy_migration.py`; audit §4 | decision on the 11 seed-pattern ratings of the one legacy account |
| Docker packaging | implemented, **not verified** | `Dockerfile`, `compose.yaml` | Docker is not installed on the delivery machine |

## 2. What the real-data evaluation shows

All numbers come from `docs/evaluation-results-real.md`, which is generated from the run directories. The brackets
are 95 % bootstrap intervals. Every run used local answer mode (`ANILOGIC_LLM__PROVIDER=none`); no paid model call
was made.

**Home recommendations (ranking protocol, 1,500 test users, K = 10).**

| Engine | Hits@10 no-rerank / greedy | Hits@10 GA | Candidate recall@20 | Catalogue coverage |
|---|---|---|---|---|
| ICF (served) | 0.062 | 0.047 | 0.087 | 0.17 |
| MF | 0.040 | 0.025 | 0.064 | 0.12 |
| NCF | 0.037 | 0.027 | 0.065 | 0.01 |
| Popularity | 0.137 | 0.129 | 0.200 | 0.005 |

* **Popularity is the strongest accuracy baseline** on this protocol, about twice ICF. The held-out positive is a
  title the user rated >= 8, and those are mostly popular titles. ICF is the best *personalised* engine and covers
  about 30 times more of the catalogue. The home page therefore shows a clearly labelled, separate "Explore" list of
  popular titles next to the personalised list.
* **GA vs greedy on the same pool:** Hits@10 −0.015 [−0.023, −0.008], diversity (ILD) +0.056, novelty +0.035, fresh
  share +0.121. The cost grew with the team package (it was −0.009 with themes for 33 titles): "fresh" rewards a
  genre or theme the user has not seen, and with themes for most titles that term now has a signal to chase. The
  validation ablation shows the novelty and the fresh terms each cost about a third of a point (full objective
  0.028, without novelty 0.032, without fresh 0.032, greedy 0.038), and MMR (λ = 0.7) reaches greedy-level
  accuracy (0.040) with more diversity (ILD 0.80 vs 0.76). The GA reaches the exact optimum of its objective in
  every one of 599 small validation pools and runs in 30 ms p50. The weights are the proposal's (0.4 / 0.3 / 0.2 /
  0.1) and were not changed; lowering the novelty and fresh weights, or MMR for the list, are product decisions the
  data supports.
* **ICF tuning:** re-run on the new version (`20261002-214030-real-homepage-81c0`); the validation choices are
  unchanged: damping 20, user-mean prior weight 5, cold-start popularity weight g = 0.9.
* **Short histories (warm-gate analysis):** with 1-30 ratings every personalised ranking stays far below popularity
  (0.12-0.14). The popularity-weighted content ranking reaches 0.031-0.062, ICF 0.016-0.032; the tag-only profile is
  close to 0. On this metric the partial-mode blend (below 8 ratings) and pure ICF (from 8 ratings) are no better
  than the content ranking; they add personalisation and coverage, which Hits@10 on popularity-biased positives does
  not reward.

**Similar titles (algorithm 1, run `20261002-215313-real-similar-7127`).** The algorithm-1 benchmark's 389
MAL-recommendation queries (weak labels: "if you liked A, try B") keep their frozen split (53 train / 102
validation / 234 test). On validation the replacement rule passes for B (logistic regression, Hits@5 0.167) and C
(BGE attribute embeddings, 0.176) against A (weighted attributes, 0.049): B − A +11.8 points [+5.9, +18.6], C − A
+12.7 [+3.9, +20.6], and B and C are tied (C − B +1.0 [−6.9, +8.8]). **B is now served in real mode** because it
ranks with A's own interpretable features, so the reasons shown do not change (`configs/real.yaml`). Test, reported
only: A 0.081, B 0.197, C 0.162; B − A +11.5 [+7.3, +16.2], and +11.9 on the 193 test queries that share no
franchise with another split. What B fixes: A normalises over the dimensions that can be compared, so a title known
only by one matching genre scores like a perfect match; K-On! got Doraemon specials from A and music / school series
from B. Franchise sanity (is a confirmed sequel / prequel in the top 5? 4,507 titles): A 0.42, B 0.64, C 0.63, the
synopsis variant 0.70.

**Collection recommendations (algorithm 2, run `real-main-01`, 3,000 test users, full catalogue of 14,182 titles,
user-macro HR@8 on rule R0).** The label is a proxy: the user's titles rated >= 8 act as the collection, one is
hidden. A0 (the slides' formula) 0.021, A1 (validation weight 0.05) 0.032, B0 (LightGBM on the same two features)
0.053, **B1 (LightGBM, eight features) 0.132**, C0 (two-tower) 0.113 (three-seed means for B and C). B1 was
pre-selected on validation and passed every pre-registered check on test: +11.1 points over A0 [+9.8, +12.3] and
+10.0 over A1 [+8.7, +11.2], AUC 0.953, no rule violations. It is the real-mode default. The team package was
imported so that the algorithm-2 catalogue stays identical (one demographic per title, no core-field changes); the
service verifies this through the catalogue hash and would fall back to the formula otherwise. Serving cost:
B1 129 ms p50 for a full-catalogue request. Details: `algorithm2_benchmark/results/real-main-01/final_report.md`.

**Rating prediction (explicit protocol, 239,937 test pairs):** MF MAE 0.872 / RMSE 1.168; NCF 0.920 / 1.241; ICF
(exact formula) 0.966 / 1.347; user mean 1.100 / 1.435 (ratings did not change with the package, nor did these
numbers). The damping that helps ICF ranking makes its rating predictions worse (MAE 1.034).

**Chat (local mode, run `20261004-170011-real-chat-4e18`).** The team package's 100 queries, written against the
package's catalogue (not human relevance judgements):

| Set | Queries | Task success | Hits@5 | Notes |
|---|---|---|---|---|
| scene | 40 | 1.00 | 1.00 | the expected scene record is among the cited evidence for all 40 (scene@3 1.00) |
| semantic (genre combinations) | 30 | 1.00 | 1.00 | every catalogue title meeting the filters counts as relevant, so this mostly checks constraint handling |
| constraint (type, score, episodes, start year, demographic, theme) | 20 | 1.00 | 1.00 | 100 % compliant |
| unanswerable | 10 | 1.00 | - | 6 impossible filters answered with no titles and the reason (e.g. "the latest start year in this snapshot is 2023"); 4 requests for data that does not exist (timecodes, rating dates, licence fees, a favourites list) answered as such |

The scene queries all target the 300 scene records, so they test finding the right one among 300. Two generated
sets test what that cannot: 60 plot questions about popular titles *without* scene records (task success 0.97, no
answer named a verified scene: before the scene rule was revised, 13 of 60 named a wrong title's scene) and 40
questions about a scene that a named title's records do not contain (no scene claimed). Synopses alone find the
title for 0.43 of the scene queries (hybrid retrieval, Hits@5), the scene layer for all of them. Other generated
sets: title lookups (exact / alias / one-typo) 0.969 over 320 queries; constraint requests 38 / 38; sequel
questions 60 / 60 (any direct sequel counts; later-released prequels are no longer "watch first"); unanswerable
scene-episode questions 30 / 30 refused with no fabricated episode number.

**Viewing plans (120 users x 5 request types):** 0 infeasible plans; GA fitness 0.80-0.83 vs greedy 0.61-0.64 on the
same objective; exact-optimal in 98.7 % of 12-unit pools (mean gap 0.0002). Prerequisites now follow release order:
among the 500 most popular titles, 28 had a later-released story prequel (Jujutsu Kaisen 0 for Jujutsu Kaisen,
No Game No Life: Zero) that the denser relation data would otherwise have scheduled first.

**System (single machine, in-process):** p50 / p95 ms — search 3.3 / 9.7, detail 6.8 / 8.8, similar 9.7 / 12.1
(cold, B) and 3.0 / 3.5 (cached), home 43.1 / 45.8 (cold) and 3.7 / 4.5 (cached), collection 128.9 / 134.5, chat
8.9 / 12.2, plan preview 232.1 / 330.1. Five concurrent readers: p95 45.8 ms, 0 errors; 90 concurrent rating writes,
0 failures. All five injected LLM failures (timeout, 429, invalid JSON, empty, refusal) fall back to local answers;
a failing dense encoder falls back to lexical search.

## 3. Default algorithms and why

| Surface | Default | Why |
|---|---|---|
| Similar titles | B: logistic regression over A's features (real mode); A: weighted attributes (synthetic mode, and the fallback) | Passed the validation replacement rule on 102 weak-label queries; tied with C there and kept for its interpretable features; confirmed on test (reported only). Chat's "something like X" uses the same scorer |
| Collection | B1 (LightGBM, eight features; real mode) / A0 (synthetic mode) | Pre-selected on validation and confirmed on test by the pre-registered rule (beats A0 and the tuned formula A1 with paired 95 % intervals above zero); falls back to the formula, with the filters still strict, if the frozen run cannot be verified. The label is a proxy (high-rating sets), not real collections |
| Home recall | ICF (damping 20, prior weight 5) | Best personalised engine on the real ranking protocol, and works for new users without retraining; MF folds in users but ranks lower; NCF cannot serve new users (ID-only) |
| Home list | GA (proposal weights) | Improves the stated list objective at a measured accuracy cost (now 1.5 points); greedy is one click away on `/dev` for comparison |
| Chat | deterministic parser + hybrid retrieval + local grounded answers | Works without credentials; an LLM adds wording only after validation. A scene is named as "verified" only when a scene record ranks first among all scene records and synopses in both retrieval channels |
| Plans | GA with exact oracle check on small pools; prerequisites in release order | Near-optimal and always feasible |

## 4. Not done / blocked

* **Human evaluation** (similarity labels, chat relevance, task-success sessions): protocol ready in
  `data/evaluation/human-evaluation-protocol.md`; not run. Status: human_evaluation_pending. The team package's
  queries and the MAL-recommendation labels are useful but are not human relevance judgements.
* **Scenes:** checked by an AI agent against the cited sources (recorded per scene in `source_ref`), not by a
  person. A human spot check is recommended before presenting them as verified facts outside this project.
* **LLM answers:** the adapters and the answer validation are implemented and tested against stubbed responses. A
  provider is configured on the delivery machine through the git-ignored `.env`, so the locally served chat calls
  it, but no evaluation run used it: every run here is local mode, and a model evaluation needs a budget approval.
  **Plain LLM vs RAG:** the Plain LLM baseline and its comparison harness are **not implemented yet**.
* **Relations for the other half of the catalogue:** not in the package; `make enrich-jikan` can still add them when
  MyAnimeList is reachable.
* **Frames, real collections:** no data; contracts, validators and pipelines are ready. The collection recommender
  is therefore selected on a proxy label; its effect on real collections is not measured.
* **Docker:** not run on the delivery machine.

## 5. Where the evidence is

`docs/test-report.md` (tests), `docs/evaluation-results-real.md` and `docs/evaluation-results-synthetic.md`
(generated summaries), `evaluation/runs/<run_id>/` (config, manifest, metrics, per-case rows, failures, report),
`algorithm2_benchmark/` (algorithm 2: pipeline, runs with protocol, models, predictions and reports, its own tests),
`data/enriched/team_package_import.json` (what was imported from the team package, with file hashes),
`docs/screenshots/` (real-mode pages) and `docs/screenshots/e2e/` (browser journeys), `var/<mode>/pipeline_runs/`
(pipeline records; local).
