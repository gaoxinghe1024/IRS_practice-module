# Test report

Final runs on the delivery machine (Apple M1 Pro, macOS 26.6, Python 3.12.14, pytest 9.1.1, Playwright 1.63 /
Chromium, ruff 0.16.9), 2026-10-01, on the final code. Raw logs are under `var/logs/` (local, not committed):
`test-final-unit-integration.log`, `test-final-e2e.log` and `lint-final.log`.

| Suite | Command | Result |
|---|---|---|
| Unit + integration, including the real-data and model checks | `make test` | **131 passed**, 0 failed, 16.7 s |
| Browser journeys 1-12 | `make e2e` | **8 passed** (12 journeys), 0 failed, 17.2 s |
| Lint | `make lint` (ruff: E, F, W, I, B, UP) | clean |
| Front-end syntax | `node --check` on every page module | clean |

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
| `unit/test_parser_conversation.py` | 11 | Numbers and negation, soft vs hard, "shorter than", typo titles, relation / plan intents, common words are not titles, contradictions ask for clarification, Chinese numbers and tags, multi-turn refinement, sequels that differ only in punctuation ("Working'!!"), titles containing commas / "!" after a cue, a typo inside such a title |
| `unit/test_llm_and_validation.py` | 9 | Provider/key gating, OpenAI-compatible success and failure outcomes, Anthropic adapter request shape and refusal handling, validation drops invented items / references / numbers, unsupported episode numbers, one repair round |
| `unit/test_chunker_security_image.py` | 5 | Chunk limits, long relation lists packed into bounded chunks of whole facts, password hashing incl. legacy PBKDF2, upload validation, uncalibrated screenshot gallery |
| `unit/test_bge_ml.py` (`ml`) | 2 | BGE vectors are 384-d and normalised, a paraphrase is closer than an unrelated sentence; chunks respect the 512-token model limit |
| `unit/test_metrics.py`, `unit/test_processed_io.py` | 3 + 2 | Hits is binary and Recall is coverage; MRR / NDCG / AUC ties / paired bootstrap; deterministic dedupe and hashing |
| `integration/test_auth_security.py` | 6 | Registration validation, cookie flags and logout, generic and throttled login errors, CSRF and cross-origin writes rejected, stable error shape, security headers |
| `integration/test_isolation.py` | 2 | A second user cannot read or change the first user's collections, chats, plans or exports; admin endpoints need admin |
| `integration/test_user_data_and_recs.py` | 9 | Idempotent ratings bump the profile version; collection semantics; progress bounds; deterministic similar; home excludes rated / excluded items and updates; negative ratings are never positive seeds; labelled cold start; strict-empty then explicit relaxation; **data and plans survive an application restart** |
| `integration/test_chat_plans_pages.py` | 6 | Chat constraints enforced and refined; no invented scene episodes or relations; LLM failure → local answer; LLM answers validated; plan budget, edits, revisions and export (including export of an edited plan); pages render and private routes are protected |
| `integration/test_pipeline_integrity.py` | 5 | Activation is idempotent, tampered artifacts are refused, a manifest mismatch degrades serving, manifests record versions and hashes, synthetic generation is reproducible |
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
| `test_j03_j04_two_collections_strict_filters_and_relaxation` | 3, 4 | `j04-collection-no-match.png`, `j04-collection-relaxed.png` |
| `test_j05_rating_cold_to_warm_and_undo` | 5 | `j05-home-personalised.png` |
| `test_j06_ga_vs_greedy_on_the_same_pool` | 6 | `j06-dev-ga-vs-greedy.png` |
| `test_j07_j08_j10_chat_constraints_refinement_local_mode` | 7, 8, 10 | `j07-chat-refinement.png` |
| `test_j09_fifteen_hour_plan_edit_and_export` | 9 | `j09-plan-edited.png` |
| `test_j12_second_account_is_isolated` | 12 | - |

Defects found by these journeys and fixed during the rebuild: the collection relaxation request carried an
extra field that the strict schema rejected (422); exporting an edited plan failed (500); `wait_for_function`
needed `eval`, which the app's CSP correctly blocks (the tests now wait on locators instead).

## 3. Real-data tests (`make test-real`)

`tests/integration/test_real_data.py` (marker `real`) runs against the activated real version `real-89bfdea00b` and
serves a throwaway copy of the business database. It is skipped automatically when no real version is active.

| Test | What it proves |
|---|---|
| `test_real_snapshot_pools_ids_and_missing_values` | Only `mal:` item keys, no duplicates; recommendable ⊆ served; recommendable titles have known scores; unknown scores stay unknown (never 0) |
| `test_real_surfaces_end_to_end` | Title search finds Steins;Gate; similar titles return 5 with reasons and exclude the target; a new account that rates 8 titles 9/10 gets a warm, ICF-explained home list whose GA fitness is not below greedy; a chat request "a comedy with at most 12 episodes and no romance" returns only titles that satisfy all three conditions (checked against the detail API); a 15-hour plan fits its budget |

Both passed in the final run, together with the two `ml` tests (`unit/test_bge_ml.py`) that load the real BGE
model.

The full real pipeline (validate → preprocess → split → train → build-index → activate) and all nine real-data
experiments were also run end to end; see `docs/evaluation-results-real.md` and `docs/operations.md` for timings.

## 4. What is not covered by automated tests

* `docker compose up --build` (no Docker on the delivery machine).
* A configured LLM provider: the adapters are tested against stubbed HTTP / SDK responses only.
* Human judgement of similarity and chat relevance (`data/evaluation/human-evaluation-protocol.md`).
