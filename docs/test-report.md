# Test report

Final runs on the delivery machine (Apple M1 Pro, macOS 26.6, Python 3.12.14, pytest 9.1.1, Playwright 1.63 /
Chromium, ruff 0.16.9), 2026-10-01, on the final code. Raw logs are under `var/logs/` (local, not committed):
`test-final-unit-integration.log`, `test-final-e2e.log` and `lint-final.log`.

| Suite | Command | Result |
|---|---|---|
| Unit + integration, including the real-data and model checks | `make test` | **131 passed**, 0 failed, 16.7 s |
| Browser journeys 1-12 | `make e2e` | **8 passed** (12 journeys), 0 failed, 17.2 s |
| Everything after the algorithm 2 work (2026-10-02, branch `yyc-algorithm2-selection`) | `pytest tests` | **148 passed** (140 unit + integration, 8 browser), 0 failed, 34 s |
| Algorithm 2 checks (experiment tests in `algorithm2_benchmark/tests/` + the website-side algorithm 2 tests) | `algorithm2_benchmark/run_checks.py` | **21 passed**, 0 failed, 49 s; output in `algorithm2_benchmark/results/tests.txt` |
| Everything after the team data package work (2026-10-04, branch `yyc-data-package`; real version `real-034ebd6a74`) | `pytest tests` | **185 passed** (177 unit + integration, including the 3 real-data and 2 BGE tests, and 8 browser tests), 0 failed, 0 skipped, 31 s; log `var/logs/test-final-data-package.log` |
| Algorithm 2 delivery files after the package import | `algorithm2_benchmark/verify_delivery.py` | **PASS**: 127 delivered files match the SHA-256 manifest |
| Lint | `make lint` (ruff: E, F, W, I, B, UP) | clean |
| Front-end syntax | `node --check` on every page module | clean |

### Algorithm 1 integration on 2026-10-10

Windows 11, Python 3.12.12, dependencies from `uv.lock` (`dev` and `llm` extras). The final-report route A is
the default in both modes; the original page templates and styles remain unchanged.

| Check | Result |
|---|---|
| `pytest tests/unit tests/integration`, followed by `pytest tests/unit/test_llm_and_validation.py` after installing the optional `llm` extra | **144 distinct cases passed**, 3 optional ML cases skipped. The base-only full run passed 142 cases; its two missing-SDK failures were resolved in the 10-case adapter retest |
| Route A regression in `unit/test_algorithm1.py` | All **1,000 targets** reproduce the published Top-5 order, scores and feature values; coverage@5 30.26%, binary hit_rate@5 67.50%, NDCG@5 0.3288 on the synthetic labels |
| Real catalogue API, account filters and fallback | Default `algorithm1_a`; target excluded; Top-5/10, hide-watched, not-interested and unavailable-scorer fallback pass |
| Playwright with Chrome, 1280 x 900 and 390 x 844 | Detail cards, recommendation reasons, 5/10/5 switching and title navigation pass; no page exceptions or horizontal overflow |
| `ruff check app pipelines evaluation tests` | clean |
| `algorithm1_benchmark/verify_delivery.py` | All 36 published file hashes match; benchmark data and predictions remain unchanged |

The unit and integration suites build a miniature synthetic world with the real pipeline commands (120 titles,
300 users; generate → preprocess → split → train → index → activate) in a temporary directory, so they never
touch the developer's data.

## 1. Unit and integration tests

| File | Tests | What it proves |
|---|---|---|
| `unit/test_cleaning.py` | 22 | Duration / aired / premiered parsing, runtime never imputed, multi-value studios with corporate suffixes, placeholder synopses, explicit-content rule, Jikan provenance counts only endpoints that returned data, supplementary files validated and mapped (invalid file → nothing imported) |
| `unit/test_similarity.py` | 4 | Two empty sets are not similar, weights redistribute over available dimensions, insufficient metadata is excluded, pairwise = single scores |
| `unit/test_constraints.py` | 6 | Missing values never satisfy hard filters, AND keyword semantics, unknown namespaces fail closed, relaxations are suggestions and never touch protected rules, the "clear filters" option, per-rule failure counts |
| `unit/test_icf.py` | 8 | Exact adjusted cosine = brute-force oracle (with and without shrinkage), centred zeros stay co-rated, co-count threshold, prediction = manual computation, low ratings contribute negatively, save/load, **uniform ratings: the exact formula gives no signal, and the user-mean prior restores it** |
| `unit/test_mf_split.py` | 2 | MF learns low-rank structure and folds in new users; the ranking split holds out distinct high ratings |
| `unit/test_ga.py` | 13 | GA reaches the exhaustive optimum on small pools, is never below the greedy seed, is deterministic per seed; crossover and mutation really change genes; the degenerate-term rules; MMR is distinct |
| `unit/test_planner.py` | 11 | Unknown durations are unplannable, GA = exhaustive and never over budget, prerequisites first, infeasible must-include reported, partial mode, feasibility checks, schedules never split movies |
| `unit/test_chat_years_studios.py` | 27 | Start-year phrases (EN / ZH, decades, ranges, "after" excludes the year), a year inside a plot, a setting period ("set from 1941 to 1945") or an age ("in their 50s") is not a limit, counts are not years; two-digit and sentence-final scores, "exactly 10", "no more than"; studio cues (strong / weak, unknown studio reported, negation); requests for data the catalogue does not hold; plot questions, tag words and quoted nicknames are not titles; year / score masks with unknown values failing, relaxation and clear-all; multi-turn year state and chips; prequels count as "watch first" only in release order |
| `unit/test_package_import.py` | 7 | The team package converter: field mapping, no genre / studio / score import, one demographic per title, synopses only from non-snapshot sources, scene provenance, **the users file is never opened**, query labels keep their source and outcome, overwrite protection, MAL-label split kept; supplement loader validation; logistic split honours a pre-registered split and audits franchise overlap; real-catalogue chat labels never reach synthetic mode |
| `unit/test_parser_conversation.py` | 11 | Numbers and negation, soft vs hard, "shorter than", typo titles, relation / plan intents, common words are not titles, contradictions ask for clarification, Chinese numbers and tags, multi-turn refinement, sequels that differ only in punctuation ("Working'!!"), titles containing commas / "!" after a cue, a typo inside such a title |
| `unit/test_llm_and_validation.py` | 9 | Provider/key gating, OpenAI-compatible success and failure outcomes, Anthropic adapter request shape and refusal handling, validation drops invented items / references / numbers, unsupported episode numbers, one repair round |
| `unit/test_chunker_security_image.py` | 5 | Chunk limits, long relation lists packed into bounded chunks of whole facts, password hashing incl. legacy PBKDF2, upload validation, uncalibrated screenshot gallery |
| `unit/test_bge_ml.py` (`ml`) | 2 | BGE vectors are 384-d and normalised, a paraphrase is closer than an unrelated sentence; chunks respect the 512-token model limit |
| `unit/test_metrics.py`, `unit/test_processed_io.py` | 3 + 2 | Hits is binary and Recall is coverage; MRR / NDCG / AUC ties / paired bootstrap; deterministic dedupe and hashing |
| `integration/test_auth_security.py` | 6 | Registration validation, cookie flags and logout, generic and throttled login errors, CSRF and cross-origin writes rejected, stable error shape, security headers |
| `integration/test_isolation.py` | 2 | A second user cannot read or change the first user's collections, chats, plans or exports; admin endpoints need admin |
| `integration/test_user_data_and_recs.py` | 9 | Idempotent ratings bump the profile version; collection semantics; progress bounds; deterministic similar; home excludes rated / excluded items and updates; negative ratings are never positive seeds; labelled cold start; strict-empty then explicit relaxation; **data and plans survive an application restart** |
| `integration/test_chat_plans_pages.py` | 7 | Chat constraints enforced and refined; no invented scene episodes or relations; questions about data the catalogue does not hold get no recommendations (timecodes, a favourites list pointing to the user's collections), an unknown studio and an impossible start year are explained; LLM failure → local answer; LLM answers validated; plan budget, edits, revisions and export (including export of an edited plan); pages render and private routes are protected |
| `integration/test_pipeline_integrity.py` | 6 | Activation is idempotent, tampered artifacts are refused, a manifest mismatch degrades serving, manifests record versions and hashes, synthetic generation is reproducible, the similar-titles default comes from the configuration and an unavailable scorer degrades to A with a status note |
| `unit/test_collection_algorithm2.py` | 6 | Served order breaks score ties by ascending id; keyword AND with case / comma normalisation, score and episode boundaries, unknown episodes, unscored and explicit titles excluded; saved titles excluded, short strict lists not padded, no match never relaxed; the profile ignores the hidden target (changing its metadata changes nothing); feature schema and NaN handling; a changed model file, catalogue hash or feature version is refused and the service falls back to A0 |
| `integration/test_collection_algorithm2.py` | 2 | The NumPy evaluator of LightGBM models (as served) reproduces native LightGBM bit for bit (NaN and zero rules, values on thresholds; `ml`); the collection API contract (strict results, no match, explicit relaxation with `strict_result_count`, scorer fallback that keeps the filters) |
| `algorithm2_benchmark/tests/` (run by `run_checks.py`) | 13 | The experiment itself: hand-computed HR@8 / NDCG@8 / AUC, tie handling, user-macro and user-level bootstrap; the whole pipeline on fixture data (user-disjoint partitions, targets never in H, training pairs only from training users, identical candidates for every model, models really fitted, LightGBM and the two-tower learn a learnable signal, service Top-8 = offline Top-8, a frozen run refuses retuning and re-evaluation reproduces, all records written) |
| `integration/test_legacy_migration.py` | 3 | Legacy ids are verified against catalogue titles (and corrected only by a unique exact title), the shipped id table is the legacy one, a dry run reports every row and writes nothing, apply takes a backup, reruns are idempotent |

Slowest steps: building the miniature world (about 4 s once per session) and loading the BGE model (about 5 s).

## 2. Browser journeys (brief section 22.3)

A live uvicorn server over the miniature world, driven with Playwright / Chromium at 1280 x 900. Seven of the eight
tests also fail on any page error or console error, so a Content-Security-Policy violation (for example an inline
style) fails them too. The isolation test deliberately requests another account's data, which is answered with a
404, so it asserts on that response instead.

| Test | Journeys | Screenshots (`docs/screenshots/e2e/`) |
|---|---|---|
| `test_j01_j11_new_user_discovery_then_not_interested` | 1, 11 | `j01-welcome.png`, `j01-home-starter.png` |
| `test_j02_detail_top5_with_sources_and_reasons` | 2 | `j02-detail-similar.png` |
| `test_j03_j04_two_collections_strict_filters_and_relaxation` | 3, 4 (incl. the strict result count after a relaxation) | `j04-collection-no-match.png`, `j04-collection-relaxed.png` |
| `test_j05_rating_cold_to_warm_and_undo` | 5 | `j05-home-personalised.png` |
| `test_j06_ga_vs_greedy_on_the_same_pool` | 6 | `j06-dev-ga-vs-greedy.png` |
| `test_j07_j08_j10_chat_constraints_refinement_local_mode` | 7, 8, 10 | `j07-chat-refinement.png` |
| `test_j09_fifteen_hour_plan_edit_and_export` | 9 | `j09-plan-edited.png` |
| `test_j12_second_account_is_isolated` | 12 | - |

Defects found by these journeys and fixed during the rebuild: the collection relaxation request carried an
extra field that the strict schema rejected (422); exporting an edited plan failed (500); `wait_for_function`
needed `eval`, which the app's CSP correctly blocks (the tests now wait on locators instead).

## 3. Real-data tests (`make test-real`)

`tests/integration/test_real_data.py` (marker `real`) runs against the activated real version (`real-034ebd6a74` since
2026-10-04) and
serves a throwaway copy of the business database. It is skipped automatically when no real version is active.

| Test | What it proves |
|---|---|
| `test_real_snapshot_pools_ids_and_missing_values` | Only `mal:` item keys, no duplicates; recommendable ⊆ served; recommendable titles have known scores; unknown scores stay unknown (never 0) |
| `test_real_team_package_layers` | With the team package imported: the frozen algorithm-2 bundle still verifies (catalogue hash unchanged), scenes are available, themes cover more than 10,000 served titles, the import report lists the users file as not read; chat honours "began airing in 2010 or later", explains an unknown studio, a start year past the snapshot and the content policy for adult tags, identifies Steins;Gate from a scene description, answers a plot question about a title without scene records at the plot level (no "verified scene"), and claims no scene for a named title asked about a scene its records do not contain |
| `test_real_surfaces_end_to_end` | Title search finds Steins;Gate; similar titles return 5 with reasons and exclude the target; a new account that rates 8 titles 9/10 gets a warm, ICF-explained home list whose GA fitness is not below greedy; a chat request "a comedy with at most 12 episodes and no romance" returns only titles that satisfy all three conditions (checked against the detail API); a 15-hour plan fits its budget |

All three passed in the final run, together with the two `ml` tests (`unit/test_bge_ml.py`) that load the real
BGE model.

The full real pipeline (validate → preprocess → split → train → build-index → activate) and all nine real-data
experiments were also run end to end; see `docs/evaluation-results-real.md` and `docs/operations.md` for timings.

The algorithm 2 stage `verify-integration` adds 14 checks on a throwaway copy of the real database for the frozen
run (`algorithm2_benchmark/results/real-main-01/integration_report.json`): API Top-8 = offline Top-8 for 1,000
requests, collection changes reach the output, strict and relaxed contracts, detail pages, and formula fallback
for a missing or changed model bundle.

## 4. What is not covered by automated tests

* `docker compose up --build` (no Docker on the delivery machine).
* A configured LLM provider: the adapters are tested against stubbed HTTP / SDK responses only.
* Human judgement of similarity and chat relevance (`data/evaluation/human-evaluation-protocol.md`).
