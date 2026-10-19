# Requirements resolution

Sources: *Intelligent Reasoning Systems Proposal for Group 10* (the report), *AniLogic-Group10-Proposal-10min*
(the slides), and the rebuild brief. Where the materials conflict or contain an inaccurate statement, this file
records the decision taken, where it is implemented and how it is checked. The historical wording stays in the
source materials. The system implements the decisions below, not the conflicting wording.

## 1. Scope

| Capability | Entry point | Method | Default output | Status |
|---|---|---|---|---|
| Similar titles (algorithm 1) | anime detail page | knowledge-based attribute matching | Top-5 (5-10 configurable) | implemented |
| Collection recommendations (algorithm 2) | collection page | strict rule filter + scorer selected by a frozen protocol (formula, LightGBM or two-tower) | Top-8 | implemented |
| Personalised home (algorithm 3) | home page | ICF recall (Top-20) + GA list optimisation | Top-10 | implemented |
| Chat-to-find (algorithm 4) | chat page | intent parsing + hybrid retrieval + grounded answer (LLM optional) | answer, title cards, sources | implemented |
| Viewing plan | plan page (also from chat) | budgeted combinatorial optimisation (GA, exact oracle for small pools) | titles, episode ranges, minutes, order, budget check | implemented |
| Relations graph, scene search, screenshot identification | detail / chat / API | data-conditional extensions | honest capability status | contract + pipeline implemented; `limited_by_data` for real data (see `docs/delivery-status.md`) |

The stack from the materials is kept: HTML/JS front end, FastAPI back end, SQLite, with recommendation and
retrieval resources prepared offline.

## 2. Conflicts and inaccurate statements

| Issue in the materials | Decision | Where implemented | Evidence |
|---|---|---|---|
| The overview stresses viewing plans, the graph and screenshots, but the MVP table lists them as extensions | The four core algorithms are delivered, plus viewing plans, explanations, cold start and feedback. The graph, scenes and screenshots are built to data contracts and enabled only when real data exists | `app/planning/`, `app/graph/`, `app/multimodal/`, `pipelines/validate_data.py` | status page; `docs/delivery-status.md` |
| Collection rules with no result "fall back to preference ranking" | The strict result stays empty. Clearly labelled relaxation *suggestions* are returned, and they apply only when the user picks one. Content, exclusion and negative-preference rules are never offered for relaxation | `app/recommendation/constraints.py` (`suggest_relaxations`) | `test_collection_rules_strict_empty_then_explicit_relaxation`; e2e journey 4 |
| HR@5 described as "how much of the ground truth is covered" | Hits@K is binary (at least one relevant item in the top K). Coverage is Recall@K and is reported separately | `evaluation/metrics.py` | `test_hits_is_binary_and_recall_is_coverage` |
| Home shows Top-10, but some experiments use a greedy Top-5 | The home page shows 10. Evaluations cut the same lists at 5 and 10, and every compared method uses the same candidate pool and final length | `evaluation/experiments/homepage.py` | homepage report (`hits@5`, `hits@10`) |
| The rating table has no timestamps, collections or watch status | No temporal order, real collections or completion status are invented. Splits are random-seeded, not temporal. The app's own new events carry timestamps | `pipelines/split.py`; `app/db/models.py` | split manifests; `docs/evaluation-protocol.md` |
| "Both titles rated highly" proposed as a similarity positive | Co-high ratings are never used as similarity truth. Supervised similarity uses human labels or a declared weak source (MAL user recommendations via Jikan) | `app/recommendation/similarity/logistic.py`, `pipelines/fit.py` | docstrings; logistic status `limited_by_data` on real data (only 2 weak-label queries) |
| NCF trained with BCE, yet all models compared on MAE | Explicit-rating regression and binary preference are separate tasks. NCF is the explicit variant (`1 + 9·sigmoid`, Huber loss), so only models that output a 1-10 rating enter the MAE/RMSE comparison | `app/recommendation/recall/ncf.py`, `evaluation/experiments/explicit.py` | explicit-rating report |
| "NCF must overfit" | Treated as a hypothesis to test. Regularisation, a validation slice and early stopping are used, and the actual result is reported | `ncf.py`, `pipelines/fit.py` | training log in the artifact manifest; explicit report |
| "Fresh ratio" undefined | Defined as the share of list items with at least one genre/theme tag not covered by the user's visible history. It is separate from popularity novelty, unknown tags are not fresh, and an empty history makes it unavailable | `app/recommendation/rerank/objective.py` | `test_novelty_unavailable_when_all_counts_zero_and_fresh_needs_history` |
| No-rerank may equal relevance-greedy Top-K | When the lists coincide, the report labels the row an alias and gives the share of users where they are identical. No artificial difference is created | `evaluation/experiments/homepage.py` | homepage report: "no-rerank == greedy for 100.0% of users" |
| Studios / demographics simplified to single values | Raw multi-valued fields remain sets. Final-report algorithm 1 A projects the alphabetically first normalised value to a scalar for exact matching; the legacy weighted scorer uses set Jaccard | `pipelines/cleaning.py`, `similarity/algorithm1.py` | `test_exact_scalar_projection_uses_primary_value_and_is_case_insensitive`; cleaning tests |
| "Score and Studios are not features" versus a five-dimension similarity that uses studio | Score is excluded from algorithm 1's similarity score. Route A uses studio exact matching and fixed weights; missing tags give 0 and unknown/nonpositive episode counts give length 0.5 | `app/recommendation/similarity/algorithm1.py` | `test_fixed_weights_missing_defaults_and_length_match_report` |
| Dense matrix described as terabytes | Memory is computed from shape x dtype. Measured on the real split: 268,600 users x 14,969 CF items x float32 = 16.1 GB (the materials' 268,599 x 14,202 is about 15.3 GB). That is gigabytes, not terabytes, and the pipeline still uses sparse CSR/CSC and never materialises the dense matrix | `pipelines/real_ingest.py` (`memory_note`), `pipelines/fit.py` (`cf`) | `preprocess_report.json`, artifact `manifest.json` |
| Differing counts (12,000 / 14,202 / 14,086 ...) | Every pool is named and counted by the pipeline: 24,905 rows in the snapshot; 23,363 served in the catalogue (1,542 excluded by the explicit-content rule); 12,429 recommendable (score and genre known); 12,313 planner-eligible; 14,969 CF items in the ranking train split | `pipelines/real_ingest.py`, `pipelines/fit.py` | `data/processed/real/<version>/reports/preprocess_report.json` |
| Specific historical numbers (37/37, 18+17+13, 14,515 chunks, 100 queries) | Kept as history in the materials only; nothing in the system hard-codes them as evidence. All reported numbers come from runs in `evaluation/runs/` | - | `docs/evaluation-results-*.md` |
| Conflicting group numbers in the source materials | The team confirmed Group 18. The display name is "Group 18" (`app.group_display`), including the configuration default; original source titles remain historical references | `configs/base.yaml`; `app/core/config.py` | page footer; `/api/v1/system/status` |
| RAG described as guaranteeing correct episode numbers and eliminating hallucination | With insufficient sources, specifics are refused ("unknown"). Retrieved evidence does not make an answer true: LLM answers are validated against the database and the retrieved documents, and unsupported specifics are removed | `app/chat/grounded_answer.py` | `test_unsupported_episode_number_rejects_answer_text`; e2e journey 8 |

## 3. Comparison scope (what "alternative" means here)

| Algorithm | Default (served) | Alternatives (runnable and evaluated unless the last column says otherwise) | Data gate |
|---|---|---|---|
| 1 Similar | Final-report A: `algorithm1_a`, fixed weights; Top-5 checked against all 1,000 published synthetic targets | Legacy weighted attributes, logistic regression, BGE attribute embeddings and tag x semantic hybrid | A needs no training; legacy B needs >= 20 train / 5 validation label queries (`logistic_min_*`) |
| 2 Collection | B1: LightGBM on eight features (real mode; A0 in synthetic mode) | A0 (the slides' formula), A1 (tuned weight), B0 (LightGBM, two features), C0 (two-tower); previous `collection_profile_v1` and `manual_slides_v0` as references | the default follows the frozen run's test confirmation (`algorithm2_benchmark/results/real-main-01/decision.json`); evaluated on a proxy label (no real collections) |
| 3 Home | ICF recall + GA | MF, NCF recall; relevance-greedy, MMR, exhaustive oracle | NCF needs the `ml` extra |
| 4 Chat | local grounded templates (plus LLM when configured) | lexical / dense / hybrid retrieval; Plain LLM vs RAG with the same LLM | Plain LLM vs RAG: comparison harness not implemented yet, and it needs an LLM key (`blocked_by_credentials` otherwise) |

Without similarity labels or real collections, a supervised model is never reported as trained on real data.
In synthetic mode every supervised model still goes through the full train, load, infer and evaluate loop.
Production paths use the algorithms that the available data supports.

## 4. Other decisions made during implementation

| Topic | Decision | Reason |
|---|---|---|
| UI language | English by default; all strings in `app/web/i18n/en.json`, so Chinese localisation can be added later | Brief section 5 |
| Chinese chat input | Numbers, tags, negation and title references are parsed; Chinese free-text plot descriptions are not matched against the English corpus, and the limitation is stated | BGE-small-en is not a cross-lingual model; no translation step is claimed |
| Kaggle `users-details` | Not imported (only offline user ids are used). Username, gender, birthday and location are never stored | Brief section 8; privacy |
| `final_animedataset.csv` | Never read; `validate-real` warns if it is present | Brief section 8 |
| Explicit content | `Hentai` / `Erotica` genres and `Rx` ratings are excluded from the served catalogue; Ecchi and R+ are not treated as explicit | Slides' rule 1, applied without over-blocking |
| Viewing-time estimates | The Duration field is a per-episode average, so plans say minutes are estimates; titles with unknown episode counts or durations are not plannable (never 12 x 24) | No imputation |
| Jikan | Offline, cached and resumable; never called while serving; current Jikan scores never overwrite the 2023 snapshot | Reproducibility, rate limits |
| ICF prediction damping and user-mean prior | Two documented parameters of the prediction (both 0 = the brief's exact formula). The prior fixes a real defect of the exact formula for new users with uniform or few high ratings (all predictions equal). Both are tuned on validation positives of a disjoint user sample only; test positives are used only for the final report, and the reports show the exact formula next to the served one | Correct new-user behaviour, leakage control |
| LLM | Off by default (`provider: none`). OpenAI-compatible HTTP or Anthropic (official SDK) when a key is set in the environment. No paid batch evaluation is run without explicit approval | Brief sections 15 and 20 |
