# API

Base path `/api/v1`, JSON in and out. The machine-readable schema is served at `/openapi.json` (interactive docs
at `/docs`), and a snapshot is committed as [`docs/openapi.json`](openapi.json). Pages under `/` are thin
shells that call this API.

## Conventions

| Topic | Rule |
|---|---|
| Authentication | `POST /auth/login` or `/auth/register` sets two cookies: `anilogic_session` (HttpOnly, SameSite=Lax) and `anilogic_csrf` (readable, SameSite=Strict). The body returns `csrf_token` too |
| CSRF | every `POST/PUT/PATCH/DELETE` by a signed-in client must send `X-CSRF-Token: <csrf_token>`. Cross-origin writes are rejected (`403 cross_origin_blocked`) |
| Ownership | private resources are visible to their owner only; another user's id answers `404 not_found` (not 403), so ids cannot be probed |
| Validation | bodies are strict: unknown fields give `422 validation_failed` |
| Errors | `{"error": {"code", "message", "details", "request_id"}}`; no stack traces, SQL or paths. The `X-Request-ID` response header carries the same id |
| Business states | "nothing matches" is **not** an error: `200` with `status`, `warnings` and, where relevant, `relaxation_options` |
| Lists | recommendation responses return `actual_k <= requested_k`; they are never padded with duplicates or ineligible titles |
| Versions | responses that depend on data carry `dataset_mode` and `dataset_version` (and `algorithm` / `model_version` in `meta`) |
| Caching | API responses are `Cache-Control: no-store`; recommendation results are cached server-side by (dataset version, `profile_version`, parameters) |

Common error codes: `not_authenticated` (401), `invalid_credentials` (401), `csrf_failed` (403),
`admin_required` (403), `not_found` (404), `username_taken` (409), `validation_failed` (422),
`must_include_invalid` (422), `rate_limited` (429), `internal_error` (500). A capability that lacks data or
configuration answers 503 with the capability status as the code (e.g. `limited_by_data`, `not_configured`).

## Endpoints

Auth: **public** = no session needed; **optional** = works signed-out, personalised when signed in; **user** =
session required; **admin** = `is_admin` account (there is no default admin; see `make-admin` in
`docs/operations.md`).

### Accounts and profile

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/register` | public | `{username, password}`; username 3-32 of `A-Z a-z 0-9 _ - .`, password >= 8 characters |
| POST | `/auth/login` | public | generic error on failure; throttled after 8 failures in 10 minutes |
| POST | `/auth/logout` | optional | clears the session |
| GET | `/me` | user | account, preferences, onboarding state, statistics |
| PATCH | `/me/preferences` | user | `{liked_tags?: [{namespace, value}], length_preference?, hide_watched_in_similar?}` |
| POST | `/me/onboarding?state=done\|skipped` | user | marks onboarding finished or skipped (the welcome page saves liked tags via `/me/preferences` and quick ratings via `/anime/{key}/rating`) |
| GET | `/me/ratings` | user | paginated rating history (`limit`, `offset`) |
| GET | `/me/progress` | user | watch progress |
| GET | `/me/exclusions` | user | "not interested" items and tags |
| GET | `/me/history` | user | saved chats and plans |

### Catalogue

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/tags?namespace=genre` | public | tag vocabulary for selects |
| GET | `/anime` | public | search / filter: `q, genres, exclude_genres, types, min_score, min_episodes, max_episodes, status, sort (relevance, popularity, score, year, title), page, page_size <= 60`; returns `rule_failures` counts |
| GET | `/anime/{item_key}` | optional | detail with unknown fields listed in `missing_fields`, provenance, sources; plus the user's rating, collections and progress when signed in |
| GET | `/anime/{item_key}/similar` | optional | algorithm 1: `k` (<= 10), `scorer`, `hide_watched`; per-item `reasons` and `components` |
| GET | `/anime/{item_key}/relations` | public | confirmed, sourced relations |

### Ratings, progress, collections, feedback

| Method | Path | Auth | Purpose |
|---|---|---|---|
| PUT / DELETE | `/anime/{item_key}/rating` | user | `{rating: 1-10}`; idempotent; bumps `profile_version` |
| PUT / DELETE | `/anime/{item_key}/progress` | user | `{status, watched_episodes}`; cannot exceed the known episode count |
| GET / POST | `/collections` | user | list / create `{name, description}` |
| GET / PATCH / DELETE | `/collections/{id}` | user | read (items + profile) / rename / delete (ratings are kept) |
| PUT / DELETE | `/collections/{id}/items/{item_key}` | user | idempotent membership |
| POST | `/collections/{id}/recommendations` | user | algorithm 2: `{keywords, include_tags, min_score, max_episodes, types, k, scorer?, relaxation_applied?: {rule_id, change, original}}`; `scorer` (A0 / A1 / B0 / B1 / C0) is for experiments and operations, the page never sends it |
| POST | `/feedback` | user | `{event_type, item_key, request_id, surface, algorithm, value, context}`; event types: click, save, unsave, rate, not_interested, undo_not_interested, helpful, unhelpful, plan_saved, relaxation_applied; `not_interested` creates an exclusion |
| GET / POST | `/exclusions` | user | list / add `{kind: item|tag, item_key? , tag?}` |
| DELETE | `/exclusions/{id}` | user | undo (revokes; the history is kept) |

### Recommendations, chat, plans

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/recommendations/home` | user | algorithm 3: `{k <= 30, engine: icf|mf|ncf, reranker: ga|greedy|none, candidate_k}`; returns `mode` (cold / partial / warm) and the list objective with the greedy baseline |
| GET / POST | `/chat/sessions` | user | list / create |
| GET / DELETE | `/chat/sessions/{id}` | user | messages and constraint state / delete |
| POST | `/chat/sessions/{id}/messages` | user | algorithm 4: `{message <= 1000 chars, save_preferences}` |
| POST | `/plans/preview` | user | compute without saving (`PlanRequest`, below) |
| GET / POST | `/plans` | user | list / compute and save |
| GET / PATCH / DELETE | `/plans/{id}` | user | read / edit (new revision, re-validated) / delete |
| POST | `/plans/{id}/reoptimize` | user | recompute with the stored request |
| GET | `/plans/{id}/export?format=json|markdown|html` | user | download (`Content-Disposition: attachment`) |

`PlanRequest`: `budget_minutes (10-20160), max_titles (1-20), mode (complete|partial), include_genres,
exclude_genres, min_score, max_episodes_per_title, diversity (low|normal|high), allow_prerequisites,
break_minutes_per_title, daily_slots [{label, minutes}], must_include [item_key], title`. `PlanUpdate`:
`title, remove_items, edit_items [{item_key, episode_start, episode_end}], order, request`.

### Data-conditional and system

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/scenes/search` | user | `{query, k}`; 503 with the capability status (e.g. `limited_by_data`) without scene data |
| POST | `/images/identify` | user | multipart `file`; size, pixel and format checked; candidates are unverified unless a threshold is calibrated |
| GET | `/system/status` | public | capability statuses for the active version |
| GET | `/system/health` | public | liveness (used by Docker) |
| POST | `/admin/reload` | admin | reload the serving state |
| GET | `/admin/runs/{request_id}` | admin | recorded recommendation run (versions, candidates, results) |
| GET | `/admin/compare/similar/{item_key}` | admin | scorers A-D side by side |
| GET | `/admin/compare/home` | admin | engines x optimisers on the same pool |

## Examples

Captured from the synthetic world (fictional titles), trimmed.

**Sign in and rate**

```http
POST /api/v1/auth/login
{"username": "demo_mystery", "password": "anilogic-demo"}
→ 200 {"user": {"id": 1, "username": "demo_mystery", "onboarding_state": "done", "is_admin": false},
       "csrf_token": "wMlN..."}

PUT /api/v1/anime/synthetic:42:094/rating        X-CSRF-Token: wMlN...
{"rating": 11}
→ 422 {"error": {"code": "validation_failed", "message": "Some fields are invalid.",
       "details": {"fields": [{"loc": ["body", "rating"], "msg": "Input should be less than or equal to 10"}]},
       "request_id": "13c99693b97d4af5"}}
```

**Similar titles** (`GET /anime/synthetic:42:094/similar?k=5`)

```json
{"items": [{"item_key": "synthetic:42:159", "title": "Hidden Carnival", "score": 0.5855,
            "reasons": [{"code": "shared_tags", "params": {"ns": "genres", "tags": "Action"}, "text": "Shares genres: Action"}],
            "components": {"coverage": 0.6, "dimensions": {"genre": 0.3333, "theme": null, "demographic": 1.0,
                                                           "studio": null, "length": 0.8465}},
            "candidate_source": "weighted_attributes", "constraints_passed": true}],
 "meta": {"requested_k": 5, "actual_k": 5, "algorithm": "weighted_attributes", "fallback": null,
          "warnings": ["This title has no themes data, so those attributes could not be compared."]}}
```

**Collection recommendations with no match** (`{"max_episodes": 1, "min_score": 9.9}`)

```json
{"status": "no_match", "items": [], "relaxed": false, "relaxed_rules": [], "strict_result_count": null,
 "relaxation_options": [{"rule_id": "filter.min_score", "change": {"min_score": 7.4}, "candidate_count": 8,
                         "reason": "Lowering the minimum score from 9.9 to 7.4 gives 8 eligible titles.",
                         "original": {"min_score": 9.9}}],
 "rule_failures": {"filter.min_score": 216, "filter.max_episodes": 177, "dedupe.excluded_items": 5},
 "meta": {"algorithm": "A0", "scorer_label": "Slides formula", "scorer_status": "ready",
          "fallback": null, "warnings": ["No titles meet all of the selected filters."]}}
```

Nothing is relaxed automatically, and a short strict list is returned as it is ("Only 4 titles meet the
filters."). To apply the option, repeat the request with `"min_score": 7.4` and
`"relaxation_applied": {"rule_id": "filter.min_score", "change": {"min_score": 7.4}, "original": {"min_score": 9.9}}`.
The response then has `"relaxed": true`, `"relaxed_rules": ["filter.min_score"]` and `"strict_result_count": 0`
(the result count under the original filters), and the page says which filter was relaxed. Items carry the eight
feature values in `components`. If the configured model bundle cannot be verified, `meta.scorer_status` is
`degraded`, the formula A0 is used and the filters stay strict.

**Home** (`POST /recommendations/home`): items carry reasons such as
`"Because you rated The Restless Labyrinth 10/10 (similarity 0.36 across 15 shared raters)"`, and the response has
`"mode": "warm"`, `"cold_start": {"usable_ratings": 12, "warm_gate": 8}` and
`"list_objective": {"fitness": 0.8076, "summary": {...}, "greedy_baseline": {"fitness": 0.7625, ...}}`.

**Chat** (`"a mystery with at most 13 episodes, no horror"`): `intent: preference_recommendation`,
`answer_mode: local_template` (no LLM configured), chips `Mystery`, `No Horror`, `<= 13 episodes`, and
`evidence` entries such as `{"doc_id": "synthetic:42:153#attr", "kind": "attributes", "source_type":
"catalog_structured", "source_ref": "...", "snippet": "..."}`. Every item has `constraints_passed: true`.

**Plan preview** (`{"budget_minutes": 900, "max_titles": 3}`): two titles, 888 of 900 minutes;
`feasibility: {"feasible": true, "problems": [], ...}`, and each item carries its `duration_evidence`
(`per_episode_average` from the catalogue Duration field). `objective` lists the optimiser (`plan_ga`), the
fitness terms, the weights and the greedy baseline.
