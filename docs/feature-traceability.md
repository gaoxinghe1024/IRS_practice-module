# Feature traceability

Every requirement from the materials, the rebuild brief (scope table, section 4) and the legacy audit, traced to
code, API, page, tests and status. Status terms follow the brief: **implemented** (real code wired to API/page),
**tested_synthetic**, **tested_real** (real inputs imported, indexed and served), **real_data_evaluated**
(evaluation run on real splits with per-case output), **limited_by_data**, **limited_by_credentials**. Test names
refer to `tests/`; `jNN` are the browser journeys in `tests/e2e/test_journeys.py`. Evaluation runs are summarised
in `docs/evaluation-results-real.md`.

## P0 — required

| Requirement | Code | API | Page | Tests | Status |
|---|---|---|---|---|---|
| Accounts: register, login, logout, sessions, per-user isolation | `app/services/auth.py`, `app/core/security.py`, `app/api/deps.py` | `/auth/*`, `/me` | `/register`, `/login` | `test_auth_security.py` (6 tests), `test_isolation.py`, j12 | implemented, tested_synthetic, tested_real |
| Catalogue: title/alias search, filters, pagination, detail with sources and missing fields | `app/services/catalog.py`, `app/retrieval/titles.py` | `GET /anime`, `/anime/{key}`, `/tags` | `/discover`, `/anime/{key}` | `test_errors_have_stable_shape_without_internals`, j01, j02; system eval (search/detail latency) | implemented, tested_synthetic, tested_real |
| Ratings 1-10, multiple collections, add/remove, watch status and progress | `app/services/user_data.py` | `/anime/{key}/rating`, `/anime/{key}/progress`, `/collections/*` | detail, `/collections`, `/profile` | `test_rating_is_idempotent_and_bumps_profile_version`, `test_collections_membership_and_deletion_semantics`, `test_progress_cannot_exceed_known_episodes`, `test_user_data_and_plans_survive_an_application_restart`, j03, j05 | implemented, tested_synthetic, tested_real |
| Algorithm 1: similar Top-5 with attribute explanations and comparisons (A-D) | `app/recommendation/similarity/*`, `services/recommend.py:similar` | `/anime/{key}/similar`, `/admin/compare/similar/{key}` | detail, `/dev` | `test_similarity.py` (4), `test_similar_excludes_target_and_is_deterministic`, j02; `similar` experiment | implemented, tested_synthetic, tested_real, real_data_evaluated (franchise sanity + weak labels; human labels pending) |
| Algorithm 2: collection Top-8, strict rules, explicit relaxation | `app/recommendation/constraints.py`, `app/recommendation/collection/*` | `/collections/{id}/recommendations` | `/collections/{id}` | `test_constraints.py` (6), `test_collection_rules_strict_empty_then_explicit_relaxation`, j03/j04; `collection` experiment | implemented, tested_synthetic, tested_real, real_data_evaluated (proxy: liked history; GBDT limited_by_data) |
| Algorithm 3: cold start, ICF Top-20 recall, GA Top-10, recall and rerank comparisons | `app/recommendation/recall/*`, `app/recommendation/rerank/*`, `services/recommend.py:home` | `/recommendations/home`, `/admin/compare/home` | `/`, `/dev` | `test_icf.py` (8), `test_mf_split.py`, `test_ga.py` (13), `test_home_excludes_rated_excluded_and_updates_with_profile`, `test_negative_ratings_are_not_used_as_positive_seeds`, `test_cold_start_user_gets_labelled_exploration`, j05, j06; `homepage`, `explicit`, `ga_ablation` experiments | implemented, tested_synthetic, tested_real, real_data_evaluated |
| Algorithm 4: language understanding, retrieval, grounded answers with sources, degradation | `app/chat/*`, `app/retrieval/*`, `services/chat.py` | `/chat/sessions/*` | `/chat` | `test_parser_conversation.py` (11), `test_llm_and_validation.py` (9), `test_chat_constraints_are_enforced_and_refined`, `test_chat_never_invents_scene_episodes_or_relations`, `test_llm_failure_falls_back_to_local_answer`, `test_llm_answer_is_validated_before_display`, j07/j08/j10; `chat` and `system` experiments | implemented, tested_synthetic, tested_real, real_data_evaluated (auto query sets; human set pending); LLM path limited_by_credentials |
| Data and validation: re-runnable import, cleaning, split, training, indexing, evaluation | `pipelines/*`, `evaluation/*` | CLI | `/status` | `test_cleaning.py`, `test_processed_io.py`, `test_pipeline_integrity.py` (5), `test_synthetic_generation_is_reproducible`; full real run recorded in `var/real/pipeline_runs/` | implemented, tested_synthetic, tested_real |

## P1 — required for the full rebuild

| Requirement | Code | API | Page | Tests | Status |
|---|---|---|---|---|---|
| Viewing plan: budget, complete / explicit partial mode, save, edit, re-optimise, export | `app/planning/*`, `services/plans.py` | `/plans/*` | `/plan`, `/plans/{id}` | `test_planner.py` (11), `test_plan_budget_edits_revisions_and_export`, j09; `planner` experiment | implemented, tested_synthetic, tested_real, real_data_evaluated |
| Explanations: attributes, collection, CF support, list diversity / novelty, plan constraints | `app/recommendation/explanations.py` | `reasons` in every list | all list pages | reason assertions in j02, `test_negative_ratings_are_not_used_as_positive_seeds`, `test_similar_excludes_target_and_is_deterministic` | implemented, tested_synthetic, tested_real |
| Cold start and feedback: short preference input, initial ratings, not-interested / exclusions, live profile updates | `services/recommend.py`, `services/user_data.py` | `/me/preferences`, `/me/onboarding`, `/feedback`, `/exclusions` | `/welcome`, cards, `/profile` | `test_cold_start_user_gets_labelled_exploration`, `test_home_excludes_rated_excluded_and_updates_with_profile`, j01 (+ journey 11), j05 | implemented, tested_synthetic, tested_real |
| History and status: own chats and plans, data version and capability availability | `app/api/v1/me.py`, `services/serving.py` | `/me/history`, `/system/status` | `/history`, `/status` | `test_pages_render_and_protect_private_routes`, `test_manifest_mismatch_degrades_serving` | implemented, tested_synthetic, tested_real |

## P2 — data-conditional extensions

| Requirement | Code | API | Tests | Status and missing data |
|---|---|---|---|---|
| Relation knowledge graph (prequel, sequel, characters) | `app/graph/relations.py`, relation / character documents | `/anime/{key}/relations`; chat `relation_query` | `test_chat_never_invents_scene_episodes_or_relations`; chat `auto_relations` set | implemented, tested_synthetic, tested_real on the partial Jikan layer; status `degraded` (relation data for 33 of 23,363 titles; elsewhere relations are unknown, not absent). Full coverage needs `data/enriched/relations.jsonl` or a completed Jikan run |
| Scene retrieval with sources | scene documents, `/scenes/search` | `/scenes/search`; chat `scene_lookup` | synthetic scene queries; `auto_no_episode` refusals on real data | implemented, tested_synthetic; **limited_by_data** on real data (`data/enriched/scenes.jsonl` with `source_ref` and `verification_status`) |
| Screenshot identification | `app/multimodal/image.py`, image index in `build_index.py` | `/images/identify` | `test_upload_validation`, `test_gallery_identify_is_uncalibrated_by_default`; `multimodal` experiment (fixtures) | implemented, tested_synthetic (pipeline only); **limited_by_data** (`data/enriched/scene_images/manifest.jsonl` + frames); no accuracy claimed |
| Two-tower collection model | `app/recommendation/collection/two_tower.py` | admin comparison | trained and evaluated on synthetic collections | implemented, tested_synthetic; **limited_by_data** (`data/evaluation/collection_events.jsonl`, >= 10,000 interactions and a validation gain) |
| Logistic similarity scorer (algorithm 1 B) | `similarity/logistic.py` | `scorer=logistic_similarity` | synthetic training and evaluation | implemented, tested_synthetic; **limited_by_data** on real data (2 weak-label queries; needs >= 20 / 5 query titles; see the human-evaluation protocol) |
| GBDT collection scorer | `collection/gradient_boosting.py` | `scorer=gbdt` | synthetic training and evaluation | implemented, tested_synthetic; **limited_by_data** (collection events) |
| LLM answers and Plain LLM vs RAG | `app/chat/llm.py`, `grounded_answer.py` | chat | `test_llm_and_validation.py` (stubbed HTTP / SDK), failure injection | LLM answer path implemented, **limited_by_credentials**; Plain LLM vs RAG comparison **not_implemented** (`ANILOGIC_LLM__PROVIDER`, `ANILOGIC_LLM_API_KEY`); local mode is the fallback |

## Legacy features kept or replaced

| Legacy feature | New location | Tests | Status |
|---|---|---|---|
| Login gate | server-side redirects and 401s | `test_pages_render_and_protect_private_routes` | implemented |
| Account quick-switch | not carried over (tokens in localStorage); see the audit | - | decided |
| Browse / search | `/discover`, `GET /anime` | j01, j02 | implemented |
| New releases list | Discover sort "Year" | API sort | implemented |
| Similar + "next five" | similar `k` up to 10 | j02 | implemented |
| Knowledge graph | relations (P2 above) | see P2 | limited_by_data |
| Ratings | rating API + history | j05 | implemented |
| Favourites and folders, move | collections (multi-membership) | j03, j12 | implemented |
| Folder hybrid recommendations | algorithm 2 (no silent relaxation) | j04 | implemented |
| Home ICF + GA with greedy comparison | algorithm 3 + `/dev` | j06 | implemented |
| Watch plan | viewing plan (P1) | j09 | implemented |
| RAG chat | algorithm 4 | j07, j08, j10 | implemented |
| Legacy scene rows | not imported (no sources) | - | decided |
| Profile and rating history | `/profile`, `/history` | j05 | implemented |
| Tag drop-downs | `/tags` feeds all selects | j01, j04 | implemented |
| Legacy accounts | `migrate-legacy` (dry run done) | `test_legacy_migration.py` | implemented |

## Acceptance journeys (brief section 22.3)

| # | Journey | Test | Result |
|---|---|---|---|
| 1 | new user's first discovery | j01 | pass |
| 2 | detail Top-5 with sources and reasons | j02 | pass |
| 3 | two collections give different profile recommendations | j03 | pass |
| 4 | strict collection filters succeed; no-match offers explicit relaxation | j04 | pass |
| 5 | rating moves cold start to personalised; undo falls back | j05 | pass |
| 6 | GA vs greedy on the same pool, term by term | j06 | pass |
| 7 | chat numeric constraints; next-turn refinement | j07 | pass |
| 8 | unsourced scene/episode question handled | j08 | pass |
| 9 | 15-hour plan: budget check, edit re-validated, save and export consistent | j09 | pass |
| 10 | LLM unavailable: local results with evidence | j10 | pass |
| 11 | not interested removes the title from recommendations | j01 | pass |
| 12 | a second account cannot see the first account's data | j12 | pass |

Browser test runs use the miniature synthetic world. The same pages were exercised against the real
activated database for the screenshots in `docs/screenshots/`.
