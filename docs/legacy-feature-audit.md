# Legacy feature audit

This audit covers the template that AniLogic replaces ("AnimeFinder", v1.4.1). It was read from the
repository at commit `426db20` (branch `main`), plus the local legacy database `backend/anime.db`, which is
not tracked by git. Everything below cites the code that was read. The rebuild happened on the branch
`anilogic-rebuild`, and the original workspace was left untouched.

The legacy code decided *what users could already do*. It did not decide the new architecture. Every
legacy capability that users relied on is either rebuilt in AniLogic or listed below with the reason it was
not carried over.

## 1. What the legacy system was

| Part | Evidence | Summary |
|---|---|---|
| Backend | `backend/main.py` (FastAPI, 25 routes), `backend/services/*.py` (8 modules), `backend/database.py` | Single file app; SQLite `backend/anime.db`; algorithms in pure Python over a 24-title hand-written seed catalogue |
| Frontend | `frontend/*.html` (6 pages) + `frontend/js/*.js` + `frontend/css/style.css` | Chinese UI, one HTML + one JS per page, shared `common.js` |
| Data | `backend/database.py` (`_seed_ratings`), `backend/load_mal.py`, `backend/jikan_enrich.py` | 24 seed titles, 675 seed ratings, 12 hand-written "scenes"; a loader for the Kaggle files existed but the app ran on the seed |
| Docs | `README.md` (v1.4.1), `DEVLOG.md` | DEVLOG rules (dated entries; one page = one HTML + one JS) are still followed |

Legacy database contents (read-only inspection, 2026-09-30): 3 users (`demo` and `visitor` have no password;
one registered account), 24 anime rows (all with `mal_id = 0`), 675 ratings (656 of them reference users
that do not exist), 8 favourites (all owned by the two seed users), 0 folders, 12 scenes.

## 2. Feature inventory and disposition

Columns follow the prompt's required format (legacy feature, user task, code evidence, legacy completeness,
valuable behaviour or asset, handling in the new system, new location, acceptance check).

| Legacy feature / page | User task | Code evidence | Legacy completeness | Valuable behaviour / asset | New system handling | New location | Acceptance |
|---|---|---|---|---|---|---|---|
| Register / login / logout | Own account | `backend/services/auth.py` (PBKDF2-SHA256 60k, token in `sessions`), `main.py:144-175` | Working; tokens passed in query strings / JSON bodies; `_resolve_uid` falls back to `user_id` or user 1 when no token | Account model; legacy hashes | Rebuilt: argon2id, HttpOnly session cookie (hash stored), CSRF double-submit, throttled generic login errors, no default admin. Legacy PBKDF2 hashes verify once and are re-hashed | `app/services/auth.py`, `app/core/security.py`, `/api/v1/auth/*`, `/login`, `/register` | `tests/integration/test_auth_security.py`; `test_password_hashing_and_legacy_verification` |
| Login gate on every page | Private pages need an account | `frontend/js/common.js` `requireAuth()` | Client-side only; APIs stayed open (see §3) | UX: private pages redirect to sign-in | Rebuilt server-side: private pages redirect to `/login?next=`; private APIs return 401 | `app/web/pages.py`, `app/api/deps.py` | `test_pages_render_and_protect_private_routes` |
| Account quick-switch (up to 8 remembered accounts) | Switch accounts quickly | `common.js` (`af_accounts` in localStorage) | Working | None worth keeping | **Not carried over.** It kept several session tokens in localStorage, where any script can read them. Sessions are now HttpOnly cookies; users sign out and in | - | Documented here |
| Browse / search (`/api/anime/list`) | Find a title | `main.py:63-76`, `explore.html/js` | Worked on 24 titles; filtering done in Python after loading every row | Search by title / Chinese title / character | Rebuilt over the full catalogue: FTS5 + aliases + fuzzy title resolution, filters (genres, excluded genres, type, minimum score, episode range, status), sorting (relevance, popularity, score, year, title), pagination; unknown values shown as unknown | `app/services/catalog.py`, `/api/v1/anime`, `/discover` | journey 1 in `tests/e2e/test_journeys.py`; `test_errors_have_stable_shape_without_internals` |
| "新剧速递" (new releases, `/api/recommend/new`) | See recent titles | `main.py:110-120` (ORDER BY aired year) | Working, no algorithm | Chronological browsing | Kept as a browse mode: Discover sort "Year" (newest first) on the 2023 snapshot; not labelled as a recommendation | `/discover?sort=year` | manual check; catalogue sort is covered by the API tests |
| Detail modal + "similar" Top-5 + "换一换" (next five) | Explore similar titles | `knowledge.py` (`0.35·Genre + 0.30·Theme + 0.15·Demo + 0.10·Studio + 0.10·Length`), `main.py:89-93` (`offset`) | Working; empty-empty dimension handling and missing fields not addressed | The five-dimension formula and weights (from the proposal) | Rebuilt as algorithm 1 (A) with availability masks (two empty sets are not similar), coverage, reasons and sources; plus B (logistic), C (BGE attribute embedding), D (tag x semantic hybrid) for comparison. Top-5 by default, `k` up to 10 replaces the offset paging | `app/recommendation/similarity/*`, `/api/v1/anime/{key}/similar`, `/anime/{key}` | `tests/unit/test_similarity.py`; journey 2 |
| Knowledge graph (`/api/anime/{id}/kg`) | See franchise / characters / studio | `knowledge.py:build_kg` (from the hand-written `franchise` field) | Worked only for the 24 seed titles | Relation view idea | Rebuilt from sourced relation data only (Kaggle has none; the partial Jikan layer gives relations for the titles fetched). Graph queries are bounded and cycle-safe; nothing is inferred from "Season 2" names. Status `limited_by_data` | `app/graph/relations.py`, `/api/v1/anime/{key}/relations` | `test_chat_never_invents_scene_episodes_or_relations`; status page |
| Rating (`POST /api/rate`, float 0-10) | Rate a title | `main.py:295-309` | Working; float ratings; ICF matrix rebuilt in-process on every rating | Ratings feed recommendations | Rebuilt: integer 1-10 (the Kaggle scale), idempotent PUT/DELETE, `profile_version` bump invalidates caches, history page | `app/services/user_data.py`, `/api/v1/anime/{key}/rating`, `/history` | `test_rating_is_idempotent_and_bumps_profile_version`; journey 5 |
| Favourites + folders (default folder, create/delete/move) | Organise titles | `main.py:181-392`; `UNIQUE(user_id, anime_id)` in `favorites` | Working; a title could live in only one folder | Multi-folder organisation | Rebuilt as named collections: a title can be in several collections, rename/delete with confirmation, deleting a collection never deletes ratings | `/api/v1/collections/*`, `/collections`, `/collections/{id}` | `test_collections_membership_and_deletion_semantics`; journeys 3 and 12 |
| Collection "hybrid reasoning" recommendations | Titles like this folder, with filters | `hybrid.py` (rules then `score/10 + 0.15·tag_hits`; **auto-relaxes silently** when rules leave nothing) | Working; silent relaxation contradicts the proposal | Rule layer + profile idea; the slides' baseline formula | Rebuilt as algorithm 2: one constraint engine (AND semantics, unknown fails closed), strict empty result plus explicit relaxation options the user must choose; scorers: `collection_profile_v1` (default), `manual_slides_v0` (the legacy formula, kept as a named baseline), GBDT, two-tower (data-gated) | `app/recommendation/constraints.py`, `app/recommendation/collection/*`, `/api/v1/collections/{id}/recommendations` | `test_collection_rules_strict_empty_then_explicit_relaxation`; journey 4 |
| Home "为你推荐" (ICF Top-20 then GA) | Personal recommendations | `cf.py` (item cosine, co-rated >= 3), `ga.py` (0.4 Rel + 0.3 Personalisation + 0.2 Div + 0.1 Nov), `icf_ga_fusion.py` | Working on 24 titles; greedy comparison shown | ICF -> GA pipeline and the greedy comparison | Rebuilt as algorithm 3: exact adjusted cosine on the 24M-rating train split, MF and NCF alternatives, cold start / warm gate, GA list optimiser (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) versus greedy / MMR / exact oracle, all evaluated | `app/recommendation/recall/*`, `app/recommendation/rerank/*`, `/api/v1/recommendations/home`, `/` | `tests/unit/test_icf.py`, `test_ga.py`; journeys 5-6; `evaluation/experiments/homepage.py` |
| Watch plan (`POST /api/watchplan`) | Fit titles into N hours | `watchplan.py` (GA, subset chromosome, repair by greedy trimming) | Working for whole titles; no progress, prerequisites, partial episodes, saving or export | The budget-GA idea | Rebuilt: complete / partial modes, watch progress, confirmed prerequisites, must-include, daily slots, save / edit / re-optimise / export, independent feasibility check after every change | `app/planning/*`, `app/services/plans.py`, `/api/v1/plans/*`, `/plan` | `tests/unit/test_planner.py`; journey 9 |
| Chat (`POST /api/chat`) incl. "名场面回忆" | Find a title by description; recall a scene | `rag.py` (token overlap retrieval, template reply, optional OpenAI-compatible call), `scenes` table (12 hand-written rows) | Working offline; LLM output shown without validation; scene answers came from unsourced rows | Offline template mode; constraint extraction ("N集内", "评分>=X") | Rebuilt as algorithm 4: deterministic parser (EN + limited ZH), multi-turn constraint state, FTS5 + dense hybrid retrieval with RRF, grounded answers with source ids, LLM optional and validated, local mode when unavailable | `app/chat/*`, `app/retrieval/*`, `app/services/chat.py`, `/api/v1/chat/*`, `/chat` | `tests/unit/test_parser_conversation.py`, `test_llm_and_validation.py`; journeys 7, 8, 10 |
| Legacy scenes data (12 rows) | Scene recall | `scenes` table: title, episode, Chinese keywords, description; no source column | Hand-written; at least one row is inconsistent (episode 1 of a film) | None that can be verified | **Not imported.** No source and no verification, and the episode numbers cannot be checked. Scene data can be supplied via `data/enriched/scenes.jsonl` with `source_ref` and `verification_status`; without it scene recall reports `limited_by_data` | `docs/data-contracts.md` | `test_chat_never_invents_scene_episodes_or_relations` |
| Profile page (`/api/user/{id}/profile`, ratings history) | See own stats | `main.py:398-450`, `profile.html/js` | Working, but readable for any user id | Stats, top genres, rating history | Rebuilt: profile (preferences, ratings with undo, watch progress, exclusions with undo) and history (saved plans and chats), both owner-only | `/api/v1/me/*`, `/profile`, `/history` | `test_second_user_cannot_see_or_change_first_users_data` |
| Tag drop-downs (`/api/tags`) | Choose tags instead of typing | `main.py:350-360` | Working | "Tags are selected, not typed" | Kept: `/api/v1/tags?namespace=...` feeds the selects (onboarding, profile preferences, Discover, collection and plan filters) | `/api/v1/tags` | journeys 1 and 4 |
| Kaggle loader | Load the real dataset | `load_mal.py` (chunked read, filters, column fallbacks) | Existed; imported `users-details` usernames; the app itself stayed on the seed | Chunked reading approach | Rebuilt as the versioned real pipeline (validate, preprocess, split, train, index, activate) with reports and manifests; usernames, gender, birthday and location are never imported | `pipelines/real_ingest.py`, `pipelines/split.py`, `pipelines/fit.py` | `test_manifests_record_versions_and_hashes`; `docs/operations.md` |
| Jikan enrichment | Covers, themes, demographics, characters | `jikan_enrich.py` (0.45 s spacing, retries, Chinese tag translation) | Worked for the 24 seed titles | Endpoint knowledge; `SEED_MAL_IDS` table | Rebuilt as an offline, resumable, cached layer (1.1 s spacing, 429 handling, per-endpoint status and SHA-256). `SEED_MAL_IDS` is kept verbatim in `pipelines/legacy_seed_mal_ids.json` for the migration only | `pipelines/jikan.py` | `test_jikan_provenance_counts_only_endpoints_that_returned_data` |
| Demo users `demo` / `visitor` and `account for test.txt` | Try the app | `database.py` seed; tracked `account for test.txt` lists four demo accounts with one shared plaintext password | Working | Demo personas | Replaced by explicit synthetic-only demo accounts created by `seed-demo` (documented in `docs/user-guide.md`, never in real mode). The tracked credential file is removed on this branch; the password stays in git history, so it must not be reused anywhere | `pipelines/demo_seed.py` | `test_register_validation_and_duplicates`; user guide |
| Algorithm labels in the UI ("算法③ · ICF + GA Fusion") | Show which algorithm produced a list | `index.html:28` etc. | Working | Transparency | Kept in a user-oriented form: each list shows its method name, candidate counts and per-item reasons; algorithm comparisons live on the admin `/dev` page | pages + `/dev` | journey 6 |

## 3. Security findings in the legacy code

These findings are why the legacy code was not wrapped or reused:

1. **No server-side authorisation on user data.** `GET /api/user/{user_id}/folders|favorites|ratings|profile` take any
   user id and need no token (`main.py:181`, `388`, `398`, `412`).
2. **Identity fallback.** `_resolve_uid` (`main.py:37-45`) accepts a caller-supplied `user_id`, or uses user 1,
   when no token is sent. Any client could therefore write ratings and favourites as any user.
3. **Tokens in URLs and bodies** (`/api/auth/me?token=`, `/api/folders/{id}?token=`), which leak into logs and history.
4. **Plaintext demo credentials in version control** (`account for test.txt`).
5. **LLM output displayed without validation** (`rag.py:_llm_reply`).

In AniLogic every one of these is covered by an automated test (`tests/integration/test_isolation.py`,
`test_auth_security.py`, `test_llm_and_validation.py`).

## 4. Data migration (legacy accounts)

`python -m pipelines.cli migrate-legacy --mode real` (dry run; `--apply` writes after taking a backup). The
tool opens the legacy database read-only, validates usernames, hash format and rating values, verifies every
legacy title id against the target catalogue, and reports every row it does not migrate.

Dry run against the activated real catalogue (2026-09-30):

| Item | Count | Outcome |
|---|---|---|
| Legacy users | 3 | 1 would migrate; 2 seed users skipped (no password), with their 5 + 3 ratings and 5 + 3 favourites counted in the report |
| Ratings of the registered account | 11 | All 11 carry the legacy seed back-fill timestamps (`2026-08-01 12:00 + id x 7 h`), so they were probably inherited from orphaned seed rows rather than entered by the user. They are listed and skipped by default; `--include-seed-pattern` migrates them |
| Orphaned ratings (user id not in `users`) | 656 | Not migratable (no owner); counted in the report |
| Favourites / folders of the registered account | 0 / 0 | - |
| Title id checks | 23 legacy ids | 22 match the MAL title of their id. `Hataraku Saibou -> 35860` is wrong (35860 is *Karakai Jouzu no Takagi-san*); the migration resolves it by unique exact title to `mal:37141` and reports the correction. `L: Change the World` is a live-action film with no MAL anime id |

In synthetic mode no legacy title can resolve (synthetic items have no MAL ids); the dry run then lists every
row as unmapped. `tests/integration/test_legacy_migration.py` covers the dry-run guarantees, the report and
idempotence.

## 5. Disposition of legacy files

Once the new system passed its tests, the legacy application was removed from this branch in one reviewed
commit, so there is only one application entry point (`app.main:app`). Files remain available in git history
(`git show 426db20:backend/main.py`).

| Path | Disposition |
|---|---|
| `backend/*.py`, `backend/services/*.py` | Removed (rebuilt in `app/`, `pipelines/`) |
| `frontend/**` | Removed (rebuilt in `app/web/`) |
| `account for test.txt` | Removed (plaintext credentials) |
| `requirements.txt` | Regenerated from `uv.lock` (pinned); no longer the legacy conda list |
| `backend/anime.db` | **Kept locally, untouched** (git-ignored user data). It is the input of `migrate-legacy`; delete it yourself only after the migration decision |
| `README.md` | Rewritten for AniLogic |
| `DEVLOG.md` | Kept; new dated entries appended |
