# Algorithm 2 repository audit

Audit date: 2026-10-02. Every number below comes from a query or a count, never from the slides. The machine-readable audit result is written to the run directory
(`audit_facts.json`) by `python -m evaluation.algorithm2.cli audit --run <run_id>`.

## 1. Repository and branch

| Item | Value |
|---|---|
| Repository | `origin` = https://github.com/gaoxinghe1024/IRS_practice-module.git |
| Target branch | `yyc-algorithm2-selection` (name chosen by the user; created from `origin/main` and already pushed to origin) |
| Starting commit | `3cda03a` (Merge pull request #4 from gaoxinghe1024/yyc-v1) |
| Remote check | `git fetch origin --prune` was run; `origin/main` = `3cda03a`, the same as the starting commit |
| Extra local changes | `var/real/anilogic.sqlite3`, `var/synthetic/anilogic.sqlite3`: runtime databases written by the service running on this machine, unrelated to this task, left as uncommitted changes and never committed |
| Project conventions | `DEVLOG.md` (every change appends a dated entry; one front-end page = one HTML plus a page JS file of the same name); no other agent instruction file (such as AGENTS.md) in the repository |

## 2. Where Algorithm 2 actually lives

| Step | File / function |
|---|---|
| Collection page | `app/web/templates/collection.html`, `app/web/static/js/pages/collection.js` (`loadRecs`, relaxation button) |
| API | `app/api/v1/collections.py::recommendations` → `POST /api/v1/collections/{id}/recommendations` (request body `CollectionRecRequest`: keywords, include_tags, min_score, max_episodes, types, k, scorer, relaxation_applied) |
| Rule filtering | `app/recommendation/constraints.py::ConstraintEngine.evaluate` (keyword, score, episodes, type, tags, de-duplication, account exclusions); `suggest_relaxations` (relaxation suggestions) |
| Preference profile | `app/recommendation/profiles.py::build_collection_profile`, `affinity` |
| Scoring | `app/recommendation/collection/manual.py`: `CollectionProfileScorer` (id `collection_profile_v1`, **the current default**), `ManualSlidesScorer` (id `manual_slides_v0`, the slides formula); `gradient_boosting.py` (sklearn HistGB, `gradient_boosting_v1`); `two_tower.py` (`two_tower_v1`) |
| Ordering / Top-K | `app/services/recommend.py::collection`: `stable_order(scores, item_key)`, then the first `k` (default 8, `configs/base.yaml: recommendation.collection.k`) |
| Relaxation / fallback | same file: when the strict result is 0 it returns `status: no_match` + `relaxation_options`, and the user re-requests with `relaxation_applied` after clicking; **there is no silent relaxation** |
| Scorer registration | `app/services/serving.py` (`state.collection_scorers`) |
| Data import | `pipelines/real_ingest.py` (Kaggle CSV → `data/processed/real/<version>/`), `pipelines/activate.py` (writes into the business database) |
| Old offline experiment | `evaluation/experiments/collection.py` (proxy benchmark: high-rating history as the collection) |
| Old training entry point | `pipelines/fit.py::_train_gbdt / _train_two_tower`: trains only when `collection_events.jsonl` exists; in real mode this is `limited_by_data` |

## 3. Whether current recommendations are really computed

- Really computed: every request reads the collection, builds the profile, applies the rules, scores and orders; there is no fixed list, hand-written result or mock.
- The "24 hand-written titles and 675 seed ratings" and "silently relaxing empty results" mentioned in the history belong to the deleted AnimeFinder template; the current `main` no longer contains that code (the template was deleted in commit `e0f1e27`), and the current implementation returns user-selectable relaxation suggestions on an empty result.
- The learned scorers (GBDT, two-tower) have **never been trained** in real mode: in `artifacts/real/real-89bfdea00b/manifest.json` both `gbdt` and `two_tower` have the status `limited_by_data` (no collection events).

## 4. Actual data (counted)

| Item | Count | Source |
|---|---|---|
| Active version | `real-89bfdea00b` | `artifacts/real/active.json` |
| Catalogue rows (snapshot) | 24,905 | `data/processed/real/real-89bfdea00b/catalog.parquet` |
| In catalogue (Hentai/Erotica/Rx excluded) | 23,363 | `in_catalog` |
| In catalogue and with a public score | **14,182** | the candidate catalogue of this experiment |
| In catalogue, with score and type (the current `recommendable` pool) | 12,429 | |
| Rating interactions (after processing) | 23,963,380 | `ratings/part-*.parquet` (columns `offline_user_id, item_idx, rating`) |
| Ratings falling on the 14,182-title catalogue | 23,954,329 | |
| Of those, personal rating ≥ 8 | 13,580,272 | |
| Users with ≥ 1 high-rated title | 264,020 | |
| Users by high-rating set size 1 / 2–5 / ≥6 | 23,055 / 38,763 / 202,202 | |
| Duplicate user-item pairs | 0 (de-duplicated during preprocessing; 0 conflicting rating pairs) | `reports/preprocess_report.json` |
| Real collections in the business database | **0** (the real database has 1 user, 0 collections, 0 ratings) | `var/real/anilogic.sqlite3` |

Conclusion: **there is no real collection data**. Algorithm 2's offline effectiveness evaluation can only use "the high-rating preference set of personal ratings ≥ 8" as a proxy label, which the reports consistently call "high-rating preference set proxy evaluation".

### 4.1 Field coverage (14,182-title catalogue)

| Field | Kaggle processed data | Teammate catalogue projection (see 4.2) | Used in this experiment |
|---|---|---|---|
| genre | 87.6% (21 tags) | 90.1% | Kaggle |
| theme | 0.2% (only 33 titles through Jikan) | **68.9%** (53 tags) | union of both |
| demographic | 0.1% | **37.2%** (5 tags; for multi-valued rows only the first in lexicographic order is kept) | union of both |
| studio | 76.2% (multi-valued, 1,042) | 78.8% (first studio only) | Kaggle |
| episodes | 99.6% | — | Kaggle |
| public score | 100% (required by the catalogue definition) | — | Kaggle |

### 4.2 The second source for theme / demographic

`algorithm1_benchmark/data/catalog.jsonl.gz` (teammate commit `d0824db`, already merged into `main`) provides a MAL theme / demographic projection for 14,202 titles, and `algorithm1_benchmark/data/provenance.json` records that its upstream is the team's official data package `AnimeFinder_system_dev_20260918_v1` (the package itself is not in the repository; its SHA-256 is recorded). The check:

- **all** of our 14,182 titles match by MAL ID;
- episodes agree 14,130/14,130, studios agree 10,799/10,812, type sets agree 12,356/12,780 (the differences come from old versus new MAL tags);
- it is the same MAL ID system, with no AniList IDs mixed in.

Handling: only theme and demographic are used as supplementary metadata for Algorithm 2 (union with the values Kaggle/Jikan already had), and `data_manifest.json` records the file hash, the coverage and the limitation that "multi-valued demographic is projected to a single value". **The shared catalogue and the business database are not modified**, so Algorithms 1, the home page and chat are unaffected. The offline experiment and the online service use the same supplement table, bound together by the catalogue hash.

## 5. Current formulas and semantics (as verified)

| Item | Current implementation |
|---|---|
| Current default scorer | `collection_profile_v1`: `0.7 × profile match + 0.3 × score/10` (not the slides formula) |
| Slides formula implementation | `manual_slides_v0`: `score/10 + 0.15 × hits` |
| Definition of hits | of a candidate's genre and theme tags, the number that **appear in the profile (frequency > 0)**; counted separately per namespace. **Frequency order is not used**, so this is not a clear implementation of "tag-frequency preference + hits" |
| Profile frequency | tag occurrences ÷ **the number of titles that have a tag in that namespace** (not ÷ \|H\|); low-rated / not-interested titles do not enter the profile |
| Tag parsing | Kaggle multi-valued fields split on commas, company suffix kept, de-duplicated; tag matching uses `norm()` (NFKC + casefold + whitespace collapse) |
| Keywords | `split_keywords`: split on `, ， 、 ; ；`, normalised, de-duplicated; each keyword must be **contained in** the type text or the title/alias text (substring match), and multiple keywords are ANDed |
| Score / episodes | `min_score` is ≥, `max_episodes` is ≤; an unknown value never satisfies either |
| Exclusions | content rules (explicit), the `recommendable` pool (scored and typed), titles already in the collection, completed titles, titles and tags the user marked "not interested" |
| Tie-breaking | score descending, `item_key` **string** ascending (`mal:10` before `mal:9`), not anime_id numeric ascending |
| Top-K | 8; when fewer than 8 exist the actual count is returned with a notice, never padded |
| Empty result | `status: no_match` + relaxation suggestions (not applied automatically) |

## 6. What this task reuses, changes and does differently

Reused:

- `ConstraintEngine` as the **only** rule implementation offline and online (keyword substring semantics, score, episodes, de-duplication, content rules); offline candidate generation and the online path call the same implementation, with a test checking equivalence.
- The Kaggle processed data (version `real-89bfdea00b`) and the MAL `item_key` system; `norm` / `split_keywords`.
- The existing relaxation-suggestion mechanism (user-triggered).

Added / changed:

- the new package `app/recommendation/algorithm2/` (catalogue, profiles, features, scorers, two-tower, serving) and `evaluation/algorithm2/` (prepare, split, train, tune, freeze, test, benchmark, report, integration check, run-all).
- `ConstraintEngine` gains the `pool.scored` rule (in catalogue and publicly scored) and a keyword mask cache; collection recommendation now uses that catalogue.
- the collection service now calls the Algorithm 2 service; the response gains relaxation status fields; the scorer is configured from the frozen conclusion.

Differences between the old behaviour and the new experiment protocol (the new protocol follows the task brief; the old implementation is kept as an auxiliary reference):

| Item | Old behaviour | New protocol |
|---|---|---|
| Candidate catalogue | `recommendable` (scored + typed, 12,429 titles) | in catalogue and scored (14,182 titles); titles with no type are kept and reported |
| Profile frequency denominator | number of titles with a tag in that namespace | \|H\| |
| hits | number of candidate tags that appear in the profile | size of the intersection between the 10 most frequent namespace-qualified tags in the profile (genre:/theme:, ties by tag lexicographic order) and the candidate's tags |
| Tie-breaking | `item_key` string | anime_id numeric ascending |
| theme / demographic | almost entirely missing | teammate MAL projection added (coverage 68.9% / 37.2%) |
| Default scorer | `collection_profile_v1` | decided by the frozen selection protocol (A0/A1/B0/B1/C0) |

The old `manual_slides_v0` (old hits) and `collection_profile_v1` (old default) are evaluated alongside as the auxiliary references `AUX_slides_v0` / `AUX_profile_v1` and do not take part in selection.

## 7. Environment

- Python 3.12.14 (uv-managed `.venv`), PyTorch 2.14.0 (MPS available), scikit-learn 1.9.1.
- New dependency: `lightgbm 4.7.0` (added to the `ml` extra in `pyproject.toml` and pinned in `uv.lock`). On macOS LightGBM needs an OpenMP runtime, so the keg-only `libomp 23.1.2` was installed with Homebrew (not linked into the global path; remove it with `brew uninstall libomp`). LightGBM wheels on Windows / Linux bundle OpenMP or use the system one.
- Hardware: Apple M1 Pro, 8-core CPU, 16 GB memory.
