# VALIDATION.md — Algorithm 4 evaluation kit: what was verified, and how

> **Scope.** Algorithm 4 evaluation: chat to find anime with an LLM, a knowledge
> base and RAG. System under test: the AnimeFinder v1.4.1 Page-4 chat pipeline,
> `backend/services/rag.py` (git blob 3b5f6dc; commit 426db20 of the team
> repository), frozen in `algo/rag_snapshot.py` with its 24-title seed knowledge
> base and 12 scenes. The AniLogic chat in `app/chat` is not part of this run:
> the method (R/N/T conditions, scoring, bootstrap CIs) carries over, the
> numbers do not.

All checks below were executed on the saved outputs; **no model was called**.

## 1. Prompt fidelity (80/80 each)

- R prompts: rebuilt from the frozen snapshot (`algo/rag_snapshot.py` +
  `algo/prompts.py`) and compared byte-for-byte with the 80 saved prompts
  (`results/extra/parity_prompts/R/`) — **80/80 identical** (tests/test_parity.py).
- N prompts: rebuilt and compared with the 80 saved prompts — **80/80 identical**.
- Retrieval: rebuilt for all 80 questions from `data/kb/anime.db` and compared
  with the saved retrieval cache (`results/retrieval.jsonl`) — **80/80 identical**.
- T answers: the template pipeline (`chat()` with LLM env vars cleared) rerun on
  all 80 queries — **80/80 identical** to the stored T replies; the stored
  answers file `results/extra/t_answers.jsonl.gz` equals the T group of
  `results/raw.jsonl.gz` (80/80).
- Snapshot vs system: `tools/verify_snapshot.py` compares every function of the
  frozen copy with the system `rag.py` (comments/whitespace stripped) — **12/12
  identical** (only the DB-path plumbing differs, marked `# SNAPSHOT-EDIT:`).

## 2. Judge aggregation replay

- Saved judge outputs (`results/judge_raw.jsonl.gz`) + saved label maps
  (`results/extra/judge_label_maps.jsonl.gz`) re-aggregated by
  `harness/judge.py --aggregate`: **960/960 per-answer `judge_avg` values
  equal** the packaged `results/scored.csv` (tolerance 1e-9; same-vendor
  exclusion included; `tests/test_tables.py` checks every judge column).

## 3. Scorer audit

- Final fact scorer (`harness/score.py:fact_ok`): replay over raw replies
  reproduces the fact-scoring results in `results/extra/fact_rescored.csv`
  **152/152** (the 48 R+ replies are scored in
  the same functions; their saved outputs live under
  `results/extra/rplus_field_completion/`).
- Title extraction (`harness/score.py:mentioned_animes`): alias table, longest
  match first, non-overlapping; aliases shorter than 3 normalized characters
  match only inside quotes; a Latin-script alias (pure-ASCII normalized form)
  also needs a word boundary in the original text — the characters just before
  and after the match are not ASCII letters or digits (CJK characters,
  punctuation, whitespace and the string ends count as boundaries). The
  boundary rule does not apply to aliases containing CJK characters or to
  exact matches inside 《》.
- Reply-level in-KB citation (`in_kb_reply_ok`): after removing echo titles
  (titles repeating the query's own) and in-KB spelling variants, no out-of-KB
  title remains and at least one work title (KB hit or out-of-KB) is
  recognized; replies that name no title do not count as in-KB citations.
- Golden cases in `tests/test_scorers.py`: English "movie" lexicon, Chinese
  numeral episodes, negation guard, frank refusals vs recommendations,
  echo/variant handling, Latin-alias word boundaries ("range(n - 1)", "Angel
  Beats!" and "Evan Call" yield no Neon Genesis Evangelion hit; NGE / EVA /
  AoT next to spaces, CJK characters or punctuation still hit), S08/S12
  exclusion set (both phrasings), T10 offtopic separation.
- Frank rate: the frank-rate definition in Appendix B reproduces the published
  per-model aggregates (`results/extra/frank_rate_by_model.csv`); agreement
  with the judge caliber is tabulated in
  `results/extra/frank_rule_vs_judge_matrix.csv`.

## 4. Replay analysis

- `harness/analyze.py` (replay, default) computes summary / RAG-lift / token /
  latency tables from packaged files only. Deterministic columns are bit-exact;
  bootstrap CIs use 10,000 resamples (seed 42) and match the published ranges
  (judge lift +0.41 to +0.67; p95 and >20 s counts identical to the report).
- `tests/test_replay.py` (fast mode) checks, per model, the R-group values
  against the published ones:

  | Model               | p95 (R) | >20 s (R) | R output tokens/call |
  | ------------------- | ------- | --------- | -------------------- |
  | Deepseek-V4.1-Flash | 8.1 s   | 0/80      | 628                  |
  | GLM-5.3             | 15.6 s  | 2/80      | 305                  |
  | Hy3                 | 19.1 s  | 4/80      | 790                  |
  | Kimi-K3             | 19.7 s  | 3/80      | 277                  |
  | GLM-5.3-Flash       | 20.5 s  | 6/80      | 313                  |
  | MiniMax-M3          | 32.4 s  | 12/80     | 528                  |

- `tests/test_replay_paths.py`: the two replay entry points (`harness/analyze.py`
  default vs `--rescore`) agree on every deterministic column (tolerance 1e-9);
  the packaged summary numbers are cross-checked against the report text verbatim
  (factoid R 19/48; frank aggregate +0.62, n = 66; in-KB citation ranges) and the
  fact/frank/in-KB rates are re-derived from the raw replies by `harness/score.py`.
- `results/scored.csv` is written by `python harness/analyze.py --rescore
  --write-scored` (rule columns from `harness/score.py` over
  `results/raw.jsonl.gz`; judge columns re-aggregated from
  `results/judge_raw.jsonl.gz`); `tests/test_tables.py` re-scores every reply
  and compares all rule columns cell by cell with the packaged file, and
  checks that `--write-scored` reproduces the file byte for byte.
- Tables 1–2 (`results/tables/table1_main_results.csv`,
  `results/tables/table2_by_type.csv`) and the derived CSVs
  `results/extra/inkb_reply_rate_per_reply.csv` and
  `results/extra/out_of_kb_per_reply.csv` are generated from
  `results/scored.csv` by `python harness/analyze.py --tables` (means exact;
  95% CI = percentile bootstrap over questions, 10,000 resamples,
  `random.Random(42)` seeded per cell). `tests/test_tables.py` regenerates all
  four with the same code and seed and compares every cell (tolerance 1e-9);
  the command's output also equals the packaged files in content.
- Figures `results/figures/fig1_grouped_bars.png` to
  `results/figures/fig4_threshold_sweep.png` (and the SVG twins) are rendered
  by `python tools/make_figures.py` from
  `results/tables/table1_main_results.csv`, `results/tables/table3_rag_lift.csv`,
  `results/tables/table4_token_usage.csv`,
  `results/extra/intent_threshold_sweep.csv` and `results/scored.csv` (the
  in-KB panels are the reply-level rate of table 1; the fig2 in-KB row is the
  paired R−N difference with the same bootstrap). `tests/test_figures.py`
  checks the plotted data against those files and, when matplotlib is
  installed, that `tools/make_figures.py` and `harness/analyze.py --figures`
  write identical, valid SVGs without timestamps or local paths.

## 5. Sensitivity analyses (included in results/extra/)

- Contaminated replies (16 leaked-markup replies excluded):
  `results/extra/sensitivity_excluding_contaminated.csv` — all lifts stay
  significant; Deepseek drops from +0.42 to +0.28 and swaps last place with Hy3.
- Disputed scene entries S08/S12 — **cluster bootstrap that resamples whole
  questions** (each draw carries all six models' replies for that question;
  10,000 resamples, seed 42; `results/extra/scene_sensitivity_cluster.csv`):
  all 22 scored scene questions (n = 132 replies per group): R 0.8561
  [0.7500, 0.9470], N 0.3864 [0.2348, 0.5530], R−N lift **+0.470
  [+0.303, +0.629]**; excluding all four disputed phrasings (18 questions,
  n = 108): R 0.8519, N 0.4259, lift **+0.426 [+0.241, +0.611]** — both lower
  bounds > 0, still significant.
- Frank aggregates (question-level cluster bootstrap,
  `results/extra/frank_aggregate_cluster.csv`): all 11 scored trap questions
  (+0.6212 [+0.3485, +0.8939]); excluding the real_not_in_kb slice (6
  questions) +0.3056 [0.0000, 0.6389] — the interval reaches 0, so the +0.31
  aggregate is **descriptive** (wide interval, six questions only); the
  real_not_in_kb slice alone +1.0000 (rule caliber).
- Pairwise significance (centered judge scores, 15 pairs, additivity checked by
  tests/test_matrix.py): `results/extra/pairwise_centered_matrix.csv`.
- Per-question named-work recall (18 questions that name a KB work by title
  or common short name; retrieval score, candidate position after
  mention-demotion, exit stage, in-prompt status):
  `results/extra/named_recall_per_question.csv` — absent from the prompt for
  11/18 (10 demoted out of the top-5 + 1 constraint-filtered; no own work row
  for 14/18 = 12 demoted + 2 filtered; 3 present only via famous-scene lines).
- False-absence declarations, per reply (`results/extra/false_absence_per_reply.csv`),
  over the 18 named questions: **43 in the R group** (fact 27, similar 16),
  **0 in the N group**; 1 of the 43 is strictly false (the named work was in
  fact present in the prompt text). The same regex also fires on 6 scene
  replies, but those are "this episode/scene is not in the KB" statements that
  already name the answer work (in the R prompt's KB text for all 6), so scene
  is outside the caliber.
- Out-of-KB residuals: `results/extra/out_of_kb_review.csv` holds one row for
  each of the 10 R replies that keep an out-of-KB title after echo and
  variant removal (read one by one: episode titles, a song title, short
  forms of an in-KB title, and works named only as suggestions to add to the
  KB with the gap stated). None is an undisclosed out-of-KB recommendation
  (0 of 480 R replies); the title extractor counts such residuals as
  out-of-KB titles, so the out-of-KB counts are an upper bound.
- Knowledge-base export: `tests/test_kb_export.py` verifies
  `data/kb/anime_kb_export.json` is field-for-field identical to
  `data/kb/anime.db` (anime 24 rows, scenes 12 rows; no user tables).

## 6. Backend tests (no real API)

- `tests/test_backends.py` starts a fake OpenAI-compatible server on
  127.0.0.1:0 and verifies request shape (single user message, temperature 0.6,
  Authorization header), HTTP-error handling (429) and timeout behavior
  (0.3 s timeout against a delayed handler). ReplayBackend and the
  documentation-only CodeBuddyCliBackend are covered too.

## 7. Docs and number consistency

- `tests/test_docs.py` cross-checks the packaged docs against the packaged
  data: REPORT Appendix D rows vs `results/extra/pairwise_centered_matrix.csv`
  and in-text "A−B = ±0.xxx" citations; Appendix E rows vs
  `results/extra/named_recall_per_question.csv`; the recall counts (18 / 7 / 4 / 3 / 11 / 10 / 1 /
  14 / 12 / 2 / 43 / 27 / 16) computed from that CSV and present verbatim in
  the text; the S08/S12 numbers vs `results/extra/scene_sensitivity_cluster.csv`;
  the README vs REPORT model tables cell-by-cell; backtick-referenced package
  paths exist; a process-narrative word list has 0 hits in README / REPORT /
  VALIDATION; results/extra/ holds one CSV per theme; the in-KB ranges quoted
  in README / REPORT (R, N, paired lift) and the R out-of-KB reply rate are
  re-derived from `results/extra/inkb_reply_rate_per_reply.csv` and
  `results/extra/out_of_kb_per_reply.csv`; the REPORT Appendix B in-KB row
  states "at least one" work title; `results/extra/out_of_kb_review.csv` has
  one row per R reply with a residual out-of-KB title; the redaction-marker
  counts quoted in README match the saved replies.
- `tests/test_tables.py` also enforces that every CSV under `results/` parses
  with the csv module (each row as wide as its header, no blank lines) and that
  the per-model R−N frank rates in table 2 equal the final / all row of
  `results/extra/frank_rate_by_model.csv` (tolerance 0.0005).
- `tests/test_docs.py` also enforces, package-wide: every file under `results/`
  (top level, tables/, figures/, extra/, with documented prefix exemptions) is
  referenced from `results/README.md` and vice versa; CSV headers carry no
  version-prefixed column names; markdown tables keep a consistent column count
  per block; the three root docs reference data files with the packaged path
  prefix; Chinese characters in markdown stay inside a documented allowlist
  (title/quote fragments only).
- Redaction: the packaged raw replies (`results/raw.jsonl.gz`) and
  `results/scored.csv` contain no workspace markers (replaced with a neutral
  placeholder); the check is automated (0 residual occurrences across all
  packaged text and .gz content) and the JSON parses cleanly after rewriting.
  The package-wide process-wording scan (`tests/test_docs.py`,
  `tools/presubmit_check.py`) reads every text file including the SVG figures,
  and every gzip file decompressed; 0 hits.

## 8. Manual spot checks

- Manual spot checks: not yet performed. Judge scores have not been compared
  with human ratings, so all judge-based results are LLM-judged only.
