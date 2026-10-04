# Evaluation protocol

How every reported number is produced. Results: [`evaluation-results-real.md`](evaluation-results-real.md) and
[`evaluation-results-synthetic.md`](evaluation-results-synthetic.md), generated from the run directories.

## 1. Principles

* **Separate protocols.** Ranking (Hits / MRR / NDCG) and explicit rating (MAE / RMSE) use different splits and
  are never merged into one table.
* **Train-only statistics.** User means, neighbours, popularity counts, novelty and model parameters come from the
  train part of the protocol in use.
* **Validation for choices, test for reporting.** Hyperparameters and default-replacement decisions use
  validation data only (ICF damping, logistic `C`, the similarity replacement rule). Test positives are touched
  once, for the final numbers.
* **Same pool, same K.** Methods compared in one table rank the same candidate universe or pool and return
  lists of the same length.
* **No invented truth.** Co-high ratings are never similarity labels; sampled negatives are "unobserved", not
  dislikes; auto-generated query sets are labelled as such and are not human relevance judgements.
* **Honest status.** An experiment that cannot run is recorded as `not_run` / `blocked_by_credentials` /
  `limited_by_data` with no numbers. Queries with an empty relevant set are reported separately, not dropped.
* **Synthetic ≠ quality.** Synthetic runs check that the implementation works; they carry a banner and are
  never quoted as recommendation quality.

## 2. Splits

| Protocol | Eligible users | Held out | Train |
|---|---|---|---|
| ranking | >= 5 ratings, including >= 2 rated >= 8 | 1 validation positive + 1 independent test positive (rating >= 8), chosen by a seeded hash | everything else |
| explicit | >= 5 ratings | random 80 / 10 / 10 per user (seeded hash) | the 80 % |

The snapshot has no timestamps, so there is no temporal split. Split manifests record a fingerprint, and every
run manifest records the fingerprints it used.

## 3. Metrics (`evaluation/metrics.py`)

For a query q with relevant set G and ranked list L:

| Metric | Definition |
|---|---|
| Hits@K | 1 if `L[:K] ∩ G` is non-empty, else 0 (binary; not a coverage ratio) |
| Recall@K | `|L[:K] ∩ G| / |G|` |
| MRR@K | 1 / rank of the first relevant item within K, else 0 |
| NDCG@K | `DCG@K / IDCG@K`, `DCG = Σ (2^rel - 1) / log2(r + 1)`; graded when labels have grades |
| AUC | Mann-Whitney probability that a positive outscores a negative (ties count 1/2) |
| MAE / RMSE | on the explicit protocol, same (user, item) pairs for every model |
| ILD | mean pairwise `1 - Sim_tag` over valid pairs of a list |
| Novelty | mean `1 - log(1 + train_count) / log(1 + max_train_count)` |
| Fresh | share of list items adding a genre/theme not in the user's visible history |
| Coverage | distinct titles recommended across users / candidate-universe size |

Uncertainty: 95 % bootstrap confidence intervals (1,000 resamples over queries/users, fixed seed). Differences
between two methods use a **paired** bootstrap over the same users; a difference whose interval excludes 0 is
called significant.

## 4. Experiments (`evaluation/experiments/`)

| Experiment | Question | Protocol and truth | Methods | Main metrics |
|---|---|---|---|---|
| `similar` | Algorithm 1: do the scorers find similar titles? | label sets kept separate: `human_annotation`, `mal_user_recommendations_weak` (real mode: the algorithm-1 benchmark's 389 MAL-recommendation queries with their frozen train / val / test split; otherwise the Jikan pairs), `synthetic_generator`, `franchise_sanity` (confirmed sequel/prequel pairs, a sanity check only); pool = recommendable titles minus the query | A weighted, B logistic (if trained), C BGE attributes, D hybrid; `semantic_synopsis` separately | Hits@5, NDCG@5, Recall@5, coverage |
| `collection` | Algorithm 2 (earlier benchmark; superseded by `algorithm2_benchmark/` (protocol, runs and conclusions in its `README.md`)): which scorer ranks the hidden collection item best under a fixed rule layer? | leave-one-out cases; real collections if provided, otherwise **`proxy_liked_history`** (a user's liked train titles act as a collection, the test positive is hidden); negatives sampled from the same eligible pool | `collection_profile_v1`, `manual_slides_v0`, GBDT (if collection events) | Hits@8, NDCG@8, AUC (full pool and 50 sampled negatives) |
| `homepage` | Algorithm 3: recall engine x list optimiser; short-history (warm-gate) analysis | ranking protocol, sampled test users (1,500 on real data); candidates = served recommendable titles not in the train history (validation positive removed); pool = each engine's top 20 | engines ICF, MF, NCF, popularity; optimisers no-rerank, greedy, GA (3 seeds); short histories (1-30 ratings): ICF served / exact, content profile, popularity, the served blend | Hits@5/10, MRR@10, NDCG@10, ILD, novelty, fresh, coverage, candidate Recall@20/50/100, GA − greedy paired differences |
| `explicit` | Which predictor rates best? | explicit protocol; models retrained on explicit-train, early stopping on explicit-validation | ICF (with coverage; with / without mean fallback), MF, NCF, user mean, item mean, global mean | MAE, RMSE, coverage |
| `ga_ablation` | What does each objective term do; how close is the GA to the optimum? | ranking **validation** positives (test positive excluded from candidates) | full / no-diversity / no-novelty / no-fresh objectives; pool 20 / 50 / 100; greedy, MMR 0.7; exhaustive oracle on 12-candidate pools (K = 5) | Hits@10, ILD, novelty, fresh, fitness; optimality gap; seed variance |
| `chat` | Algorithm 4: does chat find and constrain correctly? | labelled set `rag_queries.jsonl`, reported under its `label_source` (real mode: the team package's 100 queries, split into scene / semantic / constraint / unanswerable) or the synthetic generator's; on real data also auto sets: title lookups (exact / alias / one-typo), constraint requests, sequel questions, unanswerable scene-episode questions | lexical / dense / hybrid retrieval ablation; for scene queries the same channels over the scene corpus, the synopsis corpus and both; Plain LLM vs RAG (needs a key) | task success (TSR) by expected outcome, Hits@5, MRR@10, scene attribution (scene@1 / scene@3), constraint compliance, fabricated-episode rate, intent accuracy (only where the label names an intent), p95 latency |
| `planner` | Are plans feasible and near optimal? | offline users' train histories as profiles; 5 request types (budget, title cap, variety) | greedy density, GA (3 seeds), exact oracle on 12-unit pools | feasibility (independent re-check), fitness, utilisation, relevance, diversity, optimality gap |
| `system` | Latency and robustness | in-process app on a copy of the activated database | 30 requests per endpoint, cold/warm cache, 5 concurrent readers, 6 concurrent writers; injected LLM failures (timeout, 429, invalid JSON, empty, refusal), dense encoder failure, long / over-limit / HTML input, extreme filters, contradictions | p50 / p95 ms, error counts, pass/fail per case |
| `multimodal` | Screenshot pipeline | synthetic fixture frames only; not run on real data (no frames) | colour-histogram fixture encoder | fixture Hits@1/5 (explicitly *not* recognition accuracy) |

### Leakage controls in detail

* ICF damping (grid 0, 0.5, 2, 5, 10, 20, 50; full histories) and then the user-mean prior weight (grid 0, 2, 5,
  10, 20; histories cut to 5 / 8 / 15 ratings and the full history, mean hits@10) are tuned on the **validation**
  positives of a disjoint 20 % of eligible users (at most 600). The remaining users are evaluated on test
  positives. The chosen values are copied into the served configuration with the run id.
* The `ga_ablation` experiment uses validation positives only, and the test positive is removed from the pool.
* Collection cases rebuild the profile from the visible items only; owners are grouped so that one user's
  collections never appear in both train and test.
* Similarity label queries are split by franchise component (relation graph), so a sequel pair cannot straddle
  train and test.
* Evaluation files are never added to the retrieval corpus.

### Chat task success by expected outcome

| Expected outcome | Success means |
|---|---|
| `matches` (or answerable without an outcome) | a relevant title in the top 5, every returned title satisfies the label's hard constraints (start year, score range, episodes, type, genre / theme / demographic / studio; unknown values fail), no episode number without a verified scene record |
| `no_match` | no titles returned and no fabricated episode number |
| `insufficient_information` | no titles returned (a title the user named may be shown) and no invented specifics: episode numbers, timecodes, dates or prices |
| unanswerable without an outcome | no fabricated episode number; a contradictory request must be answered with a clarification |

The team package's semantic and constraint queries list every catalogue title that satisfies their filters as
relevant, so on those sets Hits@5 mostly measures constraint handling, not ranking quality. Its scene queries
name one scene each; `scene@3` checks that this record is among the cited evidence. These labels were written by
the team against the package's catalogue, and the scenes were checked by an AI agent, so they are reported as
`team_package_<version>`, never as human relevance judgements.

## 5. Runs and reproducibility

`python -m pipelines.cli evaluate --mode <mode> --experiment <name|all> [--sample-users N] [--seeds S]` writes
`evaluation/runs/<timestamp>-<mode>-<experiment>[-tag]-<id>/`:

| File | Content |
|---|---|
| `config.yaml` | full settings snapshot + experiment parameters |
| `manifest.json` | dataset mode and version, split fingerprints, artifact manifest hash, seeds, timings, peak memory, git commit, hardware, dependency versions |
| `metrics.json` | aggregates with CIs and sample sizes |
| `per_case.jsonl`, `predictions.csv` | per user/query rows (kept locally; git-ignored because of size) |
| `failures.jsonl` | excluded or failed cases with reasons |
| `comparison.csv` | the method table |
| `report.md` | protocol, results, limitations |

Runs are never overwritten; a rerun gets a new id. `make export-evaluation` collects the latest finished,
non-smoke run of each experiment into `docs/evaluation-results-<mode>.md` and flags newer attempts that did not
finish. `make eval-smoke` (40 users, 1 seed, tagged `smoke`) only checks the harness.

## 6. What the automatic evaluation cannot show

* Whether users perceive "similar titles" as similar: this needs human labels
  (`data/evaluation/human-evaluation-protocol.md`). The real-data similar experiment uses franchise sanity pairs
  and weak labels (MyAnimeList users' "if you liked A, try B" recommendations).
* Chat relevance as people judge it: the team package's query set has known positives and exhaustive filter
  matches, not graded human judgements of answer quality.
* Real collection behaviour: the collection experiment on real data is a proxy (liked history), not a
  collection task.
* LLM answer quality: blocked until a model key is configured. Plain LLM vs RAG: the Plain LLM baseline (same model, no retrieval) and its comparison harness are **not implemented yet**. No paid batch evaluation is
  run without approval.
* Screenshot recognition: no frame-level data.
* Online effects (click-through, satisfaction): no deployment and no timestamps; nothing is claimed.
