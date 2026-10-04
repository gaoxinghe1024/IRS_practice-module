# Data contracts

What the pipelines accept, what they produce, and how missing values are handled. Nothing below is imputed:
an unknown episode count never becomes 12, an unknown duration never becomes 24 minutes, a missing score
never becomes 0.

## 1. Real data: the Kaggle MAL 2023 snapshot

Source: Kaggle `dbdmobile/myanimelist-dataset` (licence ODbL), placed unmodified in `data/raw/mal_2023/`.

| File | Required | Columns used | Notes |
|---|---|---|---|
| `anime-dataset-2023.csv` | yes | required: `anime_id, Name, Score, Genres, Type, Episodes, Duration, Rating, Status`; optional: `English name, Other name, Synopsis, Aired, Premiered, Producers, Licensors, Studios, Source, Rank, Popularity, Favorites, Scored By, Members, Image URL` | 24,905 rows in the 2023 snapshot |
| `users-score-2023.csv` (or `.csv.gz`) | yes | `user_id, anime_id, rating` | 24,325,191 rows; read in 100,000-row chunks |
| `users-details-2023.csv` | no | headers checked only | **Not imported.** Usernames, gender, birthday and location are never stored |
| `final_animedataset.csv` | - | never read | `validate-real` warns if it is present |

`python -m pipelines.cli validate-real` checks presence, headers, sizes and SHA-256 (exit code 2 on a problem).
The raw files are never modified.

### 1.1 Field rules (`pipelines/cleaning.py`)

| Field | Rule |
|---|---|
| Unknown tokens | `"", unknown, n/a, na, nan, none, null, not available, ?, -` become `None` |
| `Genres`, `Studios`, `Producers`, `Licensors` | comma-split multi-value sets; corporate suffixes re-attached (`"Yamamura Animation, Inc."` stays one studio); case-insensitive de-duplication |
| `Synopsis` | MAL placeholder texts ("No description available for this anime." and two variants) become missing |
| `Duration` | parsed to minutes (`hr / min / sec`). With `per ep`: a per-episode average, so total = per-episode x episodes. Without `per ep` and with one episode: the runtime. Without `per ep` and several episodes: ambiguous, left unknown |
| `Aired` | ISO dates with the precision available (`1998`, `1998-04`, `1998-04-03`); `"?"` ends stay unknown |
| `Premiered` | `season year` (winter/spring/summer/fall), else unknown |
| `Score`, `Episodes`, counts | numbers with range checks; invalid values become unknown |
| `anime_id` | becomes item key `mal:<id>`. Synthetic items use `synthetic:<seed>:<n>` and never receive a MAL id |

### 1.2 Pools (counts from the current real version)

| Pool | Rule | Count |
|---|---|---|
| Snapshot rows | all parsed rows | 24,905 |
| Served catalogue | not explicit (`Hentai` / `Erotica` genre or an `Rx` rating); Ecchi and R+ stay | 23,363 (1,542 excluded) |
| Recommendable | served, score known and at least one genre or theme | 13,798 |
| Planner-eligible | recommendable, finished airing, known episodes and duration | 13,677 |
| CF items | served titles with ratings in the ranking train split | 14,969 |

Missing rates in the snapshot: score 37.0 %, studios 42.3 %, genres 19.8 %, synopsis 18.2 %, duration 3.7 %,
episodes 2.5 %. With the team package imported (§4.1) the synopsis gap falls to 14.8 % and themes cover
14,242 titles; scores, genres, studios, episodes and durations stay exactly as in the 2023 snapshot. Browsing
covers the whole served catalogue; only recommendation surfaces use the recommendable pool. The UI shows
unknown values as unknown.

Before the team package (version `real-89bfdea00b`) themes existed for 33 titles only, so the recommendable
rule required a genre (12,429 titles). With themes for most of the catalogue, a scored title with a theme but
no genre (1,369 titles, e.g. music videos and children's series) now has enough attributes to be compared and
explained, and is recommendable.

### 1.3 Ratings

Kept when the rating is an integer 1-10 and the title exists in the snapshot. Rows for explicit titles are
dropped with a count, unknown `anime_id`s are quarantined (`reports/quarantine.jsonl`), and duplicate
(user, title) pairs are resolved deterministically and counted. Current version: 24,325,191 read, 542
quarantined (unknown anime id), 361,269 dropped by the content rule, 0 duplicates, **23,963,380 kept** from
268,600 users. The dense user x title matrix would be 25.1 GB (float32, all served titles), so only sparse
CSR/CSC structures are built.

### 1.4 Splits (`pipelines/split.py`)

| Protocol | Users | Held out | Used for |
|---|---|---|---|
| `ranking` | at least 5 ratings, including at least 2 rated >= 8 | one validation positive and one independent test positive (both >= 8); the rest is train | Hits / MRR / NDCG, and training of every served model |
| `explicit` | at least 5 ratings | random 80 / 10 / 10 per user | MAE / RMSE diagnostics only |

The snapshot has no timestamps, so neither protocol is temporal and no "next watch" claim is made. Random
choices use a seeded hash of (user, item), so splits are reproducible, and each manifest records a fingerprint.
All statistics (means, neighbours, popularity counts) come from the train part only.

## 2. Processed dataset (`data/processed/<mode>/<version>/`)

| File | Content |
|---|---|
| `catalog.parquet` | one row per parsed title; **row order = catalogue index**; flags `in_catalog`, `is_recommendable`, `is_planner_eligible`, `is_explicit`; `missing_fields`; `provenance` (source, snapshot, Jikan endpoints actually used, `synopsis_source` and `supplementary_tag_sources` when a supplement filled a value) |
| `item_tags.parquet` | `item_key, namespace (genre/theme/demographic/studio/producer/licensor/source), value, source` (`source` names the supplementary provider for values merged from `tags.jsonl`) |
| `aliases.parquet` | `item_key, alias, kind` (title, English, other, synonym) |
| `relations.parquet` | `from_item, to_item, to_entity_type, to_external_id, to_name, relation_type, status, source_ref` |
| `characters.parquet` | `source_id, name, item_key, role, source_ref` |
| `scenes.parquet`, `scene_images.parquet` | only when scene data exists (see §4) |
| `ratings/part-XXX.parquet` | `offline_user_id:int64, item_idx:int32, rating:int8`, partitioned by user hash |
| `labels/*.jsonl` | derived label sets, e.g. weak MAL recommendation pairs |
| `reports/preprocess_report.json`, `reports/quarantine.jsonl` | every exclusion with a reason and count |
| `manifest.json` | mode, version, pipeline version, source hashes, config fingerprint, file checksums, dependency versions |

The version id is a hash of the source files (raw CSVs, the Jikan cache layer, any supplementary files), the
pipeline and preprocessing versions, and the relevant config. The same inputs give the same version.

## 3. Jikan cache layer (optional, `data/enriched/jikan/anime/<mal_id>.json`)

Produced by `python -m pipelines.cli enrich-jikan` (offline, resumable, 1.1 s between requests, `Retry-After`
honoured). Each file has `mal_id`, `fetched_at` and, per endpoint (`full`, `recommendations`, `characters`),
`endpoint, status, fetched_at, sha256, data`. Only endpoints that returned data count as enrichment. Used for
themes, demographics, synonyms, relations (`full`), characters, and weak similarity labels
(`recommendations`). Serving never calls Jikan, and current Jikan scores never overwrite the 2023 snapshot.

State of the current cache: 400 popular titles attempted; `/full` data for 33 titles; the other requests
failed with upstream HTTP 504 during the enrichment window (MyAnimeList behind Jikan was unavailable).

## 4. Supplementary files (optional, `data/enriched/`)

Validated by `python -m pipelines.cli validate-data`, then ingested by real preprocessing. An invalid file stops
preprocessing, and nothing is half-imported. Rows for titles outside the served catalogue are counted in the
report. One JSON object per line.

| File | Required fields | Optional fields | Effect |
|---|---|---|---|
| `tags.jsonl` | `mal_id, namespace (genre/theme/demographic/studio), value, source_ref` | `source` | values unioned with the snapshot's own values of that namespace (snapshot values first); merged before the pool flags, so they count for "recommendable" |
| `synopses.jsonl` | `mal_id, synopsis, source_ref` | `source` | fills a synopsis only where the 2023 snapshot has none; the provider is kept in `provenance.synopsis_source` |
| `relations.jsonl` | `from_mal_id, relation_type, source_ref` + one of `to_mal_id` / `to_external_id` | `to_entity_type, to_name` | relation facts (graph queries, relation documents, prerequisites; only `prequel` is a viewing prerequisite) |
| `characters.jsonl` | `mal_id, character_source_id, name, source_ref` | `role` | character documents and detail lists |
| `reviews.jsonl` | `mal_id, review_id, text, source_ref` | - | review documents (weak opinion source, never plot facts) |
| `scenes.jsonl` | `scene_id, item_key, description, source_ref, verification_status (verified/unverified)` | `episode, timestamp_seconds, characters, source_url, spoiler_level (none/minor/major)` | scene documents; enables scene search |
| `scene_images/manifest.jsonl` | `image_id, item_key, path, sha256, source_ref` | `scene_id, episode, timestamp_seconds, split` | frame gallery for screenshot identification (`path` relative to the manifest folder; files are hash-checked at index time) |

Example rows:

```json
{"from_mal_id": 16498, "to_mal_id": 25777, "relation_type": "sequel", "source_ref": "https://myanimelist.net/anime/16498"}
{"scene_id": "aot-s1-e01-wall", "item_key": "mal:16498", "description": "...", "episode": 1, "source_ref": "https://...", "verification_status": "verified", "spoiler_level": "major"}
```

Cover art is **not** a substitute for frames. With no frame manifest, screenshot identification reports
`limited_by_data`.

### 4.1 The team data package (`python -m pipelines.cli import-package`)

The team's curated package (`real.team_package_dir`, default `../Data`; see its `SYSTEM_DEV_HANDOFF.md`) starts
from the same Kaggle snapshot. `pipelines/package_import.py` converts the parts the system uses into the files
above and writes `data/enriched/team_package_import.json` (package version, SHA-256 of every file read, counts,
policies). Current import, package `20260918_A_final_v1_4a2e38e69be5`:

| Written | Rows | Titles | Notes |
|---|---|---|---|
| `tags.jsonl` | 30,737 | themes 14,242, demographics 10,395 | one demographic per title (below) |
| `synopses.jsonl` | 843 | 843 | only synopses whose provenance is not the 2023 snapshot |
| `relations.jsonl` | 20,121 | 11,334 | relation types in snake_case; duplicates removed |
| `characters.jsonl` | 40,248 | 11,494 | `character_source_id` = `<namespace>:<id>` |
| `scenes.jsonl` | 300 | 100 | all `verified`; checker, date and method kept in `source_ref`; `spoiler_level` major; 276 name an episode, none has a timecode |
| `data/evaluation/rag_queries.jsonl` | 100 | - | `label_source` `team_package_<version>` (§5) |

Policies:

* **Read only what is needed.** `DATASET_INDEX.json`, `data/canonical/anime_all.jsonl`,
  `data/rag/scenes_verified.jsonl` and `data/rag/evaluation_queries.jsonl`. `data/canonical/users.csv.gz`
  (usernames, gender) is never opened (a unit test checks this); ratings still come from the raw Kaggle files.
* **Core fields are not imported.** Genres, studios, episodes, durations and scores stay the 2023 snapshot's.
  The frozen algorithm-2 model is bound to them through its catalogue hash, and mixing 2026 values into a 2023
  snapshot would make the data inconsistent.
* **One demographic per title**, the alphabetically first where the package lists several (46 titles). This is
  the rule of the team's algorithm-1 projection that algorithm 2 was trained with; with it, the new snapshot
  gives the identical algorithm-2 catalogue hash (`24c83192...`), and the projection adds no value the snapshot
  lacks, so both algorithms see one catalogue.
* Records whose MAL identity the package did not confirm (114) are skipped. The package must be marked
  `final`. An existing `rag_queries.jsonl` from another label source is not replaced without `--force`.
* **Scenes were checked by an AI agent** against cited sources (recorded per scene), not by a person. A human
  spot check is recommended before they are presented as verified facts outside this course project.

## 5. Evaluation files (optional, `data/evaluation/`)

Templates with one example each are in `data/evaluation/templates/`. The protocol for producing human labels is
`data/evaluation/human-evaluation-protocol.md`. Evaluation files are never written into the retrieval corpus.

| File | Required fields | Used by |
|---|---|---|
| `similarity_labels.jsonl` | `query_item, positives: [{item_key, grade 1-3}], label_source` | algorithm 1 B training (grouped train/val/test by franchise) and evaluation |
| `similarity_labels_mal.jsonl` | `query_item, positives: [{item_key, grade, votes}], label_source, split (train/val/test)` | weak labels from MyAnimeList users' recommendations (`import-similarity-labels` from `algorithm1_benchmark/data/mal_ground_truth.jsonl`, 389 queries); used by algorithm 1 B in real mode instead of the processed weak pairs, with the benchmark's pre-registered split kept and franchise overlap across splits reported |
| `collection_events.jsonl` | `collection_id, offline_user_id, items: [item_key...]` (>= 3 items per collection for leave-one-out) | algorithm 2 GBDT / two-tower training and evaluation |
| `rag_queries.jsonl` | `query_id, query, intent (may be null: no single expected intent), relevant_item_keys, answerable` (+ optional `hard_constraints: {max_episodes, min_episodes, min_score, max_score, year_from, year_to, types, include_/exclude_ genres, themes, demographics, studios}`, `expected_outcome (matches/no_match/insufficient_information)`, `expected_scene_ids`, `category`, `label_source`) | chat evaluation, labelled set; reported under its `label_source` and split by `category` |
| `scene_queries.jsonl` | `query_id, query, target_item_key` (+ `target_scene_id, episode`) | scene search evaluation |
| `image_queries.jsonl` | `query_id, path, sha256, in_gallery` (+ `target_item_key`) | screenshot evaluation (in-gallery and out-of-gallery) |
| `system_tasks.jsonl` | `task_id, surface, steps, success` | recorded human task-success sessions |

## 6. Synthetic mode

`python -m pipelines.cli build-all --mode synthetic` generates a seeded fictional world (`data/synthetic/`):
200 titles, 800 users and about 20,000 ratings by default (`configs/synthetic.yaml`), plus relations,
characters, scenes, fixture frames and label files. Its latent structure gives CF / MF / NCF something to learn.
Everything is labelled `synthetic`. Synthetic numbers validate the implementation and never stand for real
recommendation quality.
