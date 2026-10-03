# Algorithm 4 — Chat to Find Anime (LLM + KB + RAG): Evaluation Kit

> **Scope.** Algorithm 4 evaluation: chat to find anime with an LLM, a knowledge  
> base and RAG. System under test: the AnimeFinder v1.4.1 Page-4 chat pipeline,  
> `backend/services/rag.py` (git blob 3b5f6dc; commit 426db20 of the team  
> repository), frozen in `algo/rag_snapshot.py` with its 24-title seed knowledge  
> base and 12 scenes. The AniLogic chat in `app/chat` is not part of this run:  
> the method (R/N/T conditions, scoring, bootstrap CIs) carries over, the  
> numbers do not.

## Question

Does retrieval-augmented generation (RAG) improve AnimeFinder's Page-4  
"chat-to-find-anime" feature, and which of the six available LLMs should the  
system use — and should the template pipeline (T) be kept at all? The evaluation  
compared 80 test queries (74 Chinese, 6 English) across six question types  
(scene 24 / constraint 16 / trap 12 / similar 10 / mood 10 / factoid 8) under  
three conditions: R (RAG prompt built from the production retrieval pipeline),  
N (LLM only, no retrieval), and T (the production template pipeline).

## Compared options

- **R** — production `rag.py` retrieval (lexical scoring, scene-intent threshold  
  0.34) feeding retrieved KB text into the prompt (frozen copy in  
  `algo/rag_snapshot.py`). The system's real LLM request (temperature 0.6, no  
  `max_tokens`, 20 s timeout) was not in effect here — see REPORT.md,  
  "Scope of what was executed".
- **N** — same six models, role line only, no retrieval.
- **T** — the real template pipeline (`chat()` with LLM env vars cleared). T is  
  kept in this kit for reproducibility, even if the system later drops template  
  replies.
- **Models** (display ↔ CLI id): Hy3 ↔ `hy3`; Deepseek-V4.1-Flash ↔  
  `deepseek-v4.1-flash`; GLM-5.3 ↔ `glm-5.3`; GLM-5.3-Flash ↔ `glm-5.3-flash`;  
  MiniMax-M3 ↔ `minimax-m3`; Kimi-K3 ↔ `kimi-k3-1`. 160 calls per model  
  (80 questions × R/N), plus 80 template outputs for T and 240 judge calls.

## Data

- `data/testset_v1.jsonl` — the 80 test queries. **The test set is Chinese-first  
  (74/80) because the Page-4 knowledge base and alias tables carry Chinese names  
  and famous-scene descriptions; the 6 English questions are analyzed separately.**
- `data/aliases.json` — alias table for the 24 KB titles (137 aliases).
- `data/kb/anime.db` — the SQLite knowledge base used by Page-4: the `anime` (24  
  titles) and `scenes` (12 famous scenes) tables of the system database, rows  
  unchanged; the system's user, rating, favourite and session tables are not  
  included. `data/kb/anime_kb_export.json`  
  is a human-readable, field-for-field equivalent export of the same content.
- `results/` — saved raw outputs (raw.jsonl.gz = T/N/R × 6 models), judge  
  outputs, retrieval cache, per-reply metrics, tables, figures.

## Metrics

Judge score (1–5; 3 LLM judges, listwise, same-vendor judge excluded; per-judge  
mean-centering as the main caliber), KB-consistent episode accuracy, all-constraints-  
satisfied, per-work constraint rate, reply-level in-KB citation rate (echo titles and  
in-KB variants removed; no out-of-KB title remains and at least one work title is  
recognized), frank rate (declaring missing information instead of fabricating;  
T10 offtopic reported separately), fact accuracy (final scorer: EN+ZH keyword  
lexicon, Chinese-numeral fallback, negation guard), token usage and latency.  
Full definitions: `REPORT.md` Appendix B.

## Results at a glance (R group, centered judge caliber)

| Model               | Judge (raw) | Centered (rank) | >20 s | p95    | Output tokens/call |
| ------------------- | ----------- | --------------- | ----- | ------ | ------------------ |
| GLM-5.3-Flash       | 4.08        | 0.45 (1)        | 6/80  | 20.5 s | 313                |
| GLM-5.3             | 4.07        | 0.44 (2)        | 2/80  | 15.6 s | 305                |
| Kimi-K3             | 4.12        | 0.43 (3)        | 3/80  | 19.7 s | 277                |
| MiniMax-M3          | 4.02        | 0.33 (4)        | 12/80 | 32.4 s | 528                |
| Deepseek-V4.1-Flash | 3.99        | 0.30 (5)        | 0/80  | 8.1 s  | 628                |
| Hy3                 | 4.03        | 0.28 (6)        | 4/80  | 19.1 s | 790                |

RAG lift: judge +0.41 to +0.67; episode accuracy +0.36 to +0.64; in-KB citation  
(reply level) +0.59 to +0.80; the only significant regression is factoid  
(−0.50 to −0.62), entirely caused by the named work missing from the prompt text.

## Findings for the evaluated pipeline

1. **Keep RAG on Page 4** *(verified)* — broad judge/hard-metric gains at ≈+400  
   net input tokens per query. Until named-work recall is fixed, **do not route  
   factoid questions through RAG** (see REPORT.md Recommendations 1/3).
2. **Model choice is scenario-conditional** *(numbers verified; the choice is  
   judgment)* — the centered ranking is two-tier: GLM-5.3-Flash / GLM-5.3 / Kimi  
   significantly above MiniMax / Deepseek / Hy3 (9 of 15 pairwise gaps, all  
   cross-tier; no separation inside either tier). Quality-first:  
   GLM-5.3-Flash / GLM-5.3; latency-critical: Deepseek; stable quality + lowest  
   output tokens: Kimi-K3. Hy3 sits in the same bottom tier as Deepseek and  
   MiniMax (statistically tied, no quality edge) and has the highest output  
   tokens per call (790) — see REPORT.md Recommendation 2 for the full facts.
3. **Fix named-work recall** *(root cause verified; the fix itself is not)* —  
   18 of the 80 queries name a KB work by title or common short name (8 factoid,  
   10 similar; the 6 scene questions match only generic nouns or character  
   names and are not counted as naming). The named work is absent from the  
   R prompt's KB text for 11 of 18 (fact 5, similar 6): 10 of the 11 misses  
   are pushed out of the top-5 candidates by mention-demotion — intentional  
   design for similar queries, a defect for fact/scene — and 1 similar question  
   is removed by the constraint filter. Present queries: 7/18 = 4 with an own  
   work-entry row + 3 only via famous-scene lines (per-question table  
   `results/extra/named_recall_per_question.csv`).
4. **Keep the template pipeline only as an offline fallback** *(verified as a  
   lower bound)* — it needs a no-match/low-confidence branch; it cannot refuse.

Full reasoning and caveats: `REPORT.md`.

## Caveats

- Single run (repetition 1); famous-titles KB (ceiling effects); LLM judges with  
  no human ground truth; judge centering changes the ranking; length and  
  faithfulness dimensions favor R; two KB scene entries (S08/S12, each with two  
  phrasings) are suspected to differ from the real animations — the sensitivity  
  analysis excluding all four questions is included and the lift remains  
  significant (+0.426 [+0.241, +0.611]; cluster bootstrap resampling whole  
  questions, `results/extra/scene_sensitivity_cluster.csv`).
- Absolute scores, latencies and timeout rates were produced through the CLI  
  channel and should not be read as end-to-end results for the deployed system  
  with a vendor key — see "Scope of what was executed" at the top of REPORT.md.
- **Live backends were only exercised against a fake OpenAI-compatible server on  
  127.0.0.1 (tests/test_backends.py). No vendor API was ever called from this  
  package.**

## How to reproduce

```bash
pip install -r requirements.txt          # 1) install (stdlib-only by default)
python -m unittest discover -s tests -v  # 2) run all tests (offline, < 5 min)
python harness/analyze.py                # 3) replay analysis -> results/replay/
```

Replay mode never needs credentials or network; it uses the Python standard  
library only (tested with Python 3.10 on Windows 11; matplotlib 3.8.0 is used  
for the figures alone). Live generation (explicitly  
opt-in): `python harness/run_generation.py --group R --model <model> --backend
openai` with `LLM_API_BASE=<OpenAI-compatible base URL>` and `LLM_API_KEY=<your  
key>`. `--model` accepts either a display name from `--list-models` (mapped to
its CLI id) or a raw model id (used as-is when no display name matches).
`LLM_MODEL` is the fallback model id for the OpenAI-compatible backend, used
only when a call does not name a model explicitly. Accepted request parameters
differ by vendor.

Where the numbers come from:

| Numbers | Source |
| --- | --- |
| Per-model/group metric rates, judge averages (results/replay/ summary tables) | `python harness/analyze.py` — default replay over `results/scored.csv` |
| Rule metrics re-scored from the raw replies (consistency check; must match the default replay) | `python harness/analyze.py --rescore` — re-scores `results/raw.jsonl.gz` via `harness/score.py` |
| Judge aggregation and the pairwise centered matrix | `python harness/judge.py --aggregate` and `python harness/analyze.py --pairwise-matrix` — from `results/judge_raw.jsonl.gz` + `results/extra/judge_label_maps.jsonl.gz` |
| Tables 1–2 (`results/tables/table1_main_results.csv`, `results/tables/table2_by_type.csv`) and the derived CSVs `results/extra/inkb_reply_rate_per_reply.csv`, `results/extra/out_of_kb_per_reply.csv` | `python harness/analyze.py --tables` — from `results/scored.csv` (95% CI: percentile bootstrap over questions, 10,000 resamples, `random.Random(42)` per cell) |
| Per-reply scores (`results/scored.csv`) | `python harness/analyze.py --rescore --write-scored` — rule columns from `harness/score.py` over `results/raw.jsonl.gz`, judge columns re-aggregated from `results/judge_raw.jsonl.gz` |
| Figures (`results/figures/`, fig1–fig4 as PNG + SVG) | `python tools/make_figures.py` (or `python harness/analyze.py --figures`) — from `results/tables/table1_main_results.csv`, `results/tables/table3_rag_lift.csv`, `results/tables/table4_token_usage.csv`, `results/extra/intent_threshold_sweep.csv` and `results/scored.csv`; needs matplotlib |
| Tables 3–8 and the other files in results/extra/ | given as-is; produced by the team's offline analysis scripts, not by the replay pipeline |

## Layout

```
README.md  REPORT.md  VALIDATION.md  CHANGELOG.md  MANIFEST.txt  requirements.txt
algo/      rag_snapshot.py (frozen Page-4 pipeline) · prompts.py · llm_backend.py
data/      testset_v1.jsonl · aliases.json · kb/anime.db · kb/anime_kb_export.json
harness/   models.yaml · run_generation.py · score.py · judge.py ·
           analyze.py (replay · --rescore · --tables · --figures)
tools/     verify_snapshot.py (snapshot vs system diff) · presubmit_check.py ·
           make_figures.py (fig1-fig4 from the packaged data; needs matplotlib)
tests/     test_parity · test_scorers · test_replay · test_replay_paths · test_tables ·
           test_figures · test_backends · test_matrix · test_kb_export · test_docs
           (unittest)
results/   raw.jsonl.gz · judge_raw.jsonl.gz · retrieval.jsonl · scored.csv ·
           tables/ · figures/ · extra/ (R+ experiment, sensitivity analyses,
           pairwise matrix, per-question recall table, out-of-KB review, token
           usage, judge label maps, parity prompts) · README.md (file map)
```

## Redaction

- `<profile>` inside saved replies marks the local user-profile directory name  
  that leaked into 16 agent-tool payloads; the surrounding paths belong to the  
  evaluation harness, not to any reply content.
- `<workspace>` marks the evaluation workspace directory; it replaces the local  
  parent-directory prefix and the project-name fragment wherever both appeared.  
- `<pkg>` marks the evaluation package folder name: it follows `<workspace>/` in  
  the 48 `prompt_file` paths of `results/extra/rplus_field_completion/replies.jsonl.gz`  
  and stands for a project-directory name in one saved reply in  
  `results/raw.jsonl.gz`.

## Provenance

- System: AnimeFinder backend v1.4.1, `backend/services/rag.py` (266 lines),  
  snapshot copied 2026-09-30 and verified function-by-function identical  
  (`tools/verify_snapshot.py`: 12/12).  
  The system file is available in the team repository with  
  `git show 426db20:backend/services/rag.py`.
- Dates: main experiment 2026-09-29; R+ field-completion follow-up 2026-09-30;  
  all statistics are computed from the saved raw outputs.
- Scope: the system code was run as a Python module, not as the deployed  
  service, and all model answers came through the CLI channel — see "Scope of  
  what was executed" at the top of REPORT.md.
- Channel: CodeBuddy CLI 2.158.0 (headless), single-turn, prompt via stdin; the  
  exact invocation is documented in `algo/llm_backend.py`  
  (`CodeBuddyCliBackend`). Hidden CLI overhead ≈2,976 input tokens per call  
  (estimated by OLS regression; R² ≥ 0.998 for 5 of 6 models, unreliable for  
  MiniMax-M3 — its output-token column is a direct reading and is fine).
- `CodeBuddyCliBackend` requires a WorkBuddy/CodeBuddy login and cannot run  
  elsewhere; it is documentation-only.

## Data note

The knowledge base (`data/kb/anime.db`: 24 titles in `anime`, 12 scenes in  
`scenes`) is the hand-written demo seed of the AnimeFinder v1.4.1 backend  
(`ANIME_SEED` and `SCENE_SEED` in its `backend/database.py`); the 24 anime rows and  
12 scene rows match those two seed lists field for field. `mal_id` is 0 and  
`image_url` is empty for every row, and no MyAnimeList, Jikan or Kaggle rows or  
images are included. The file holds no credentials.  
`data/kb/anime_kb_export.json` is a directly readable equivalent export of the  
same content (anime.db is a SQLite binary). Reuse terms for the knowledge base  
and for the frozen `algo/rag_snapshot.py` are not stated in this folder; they are  
for the repository owner to set.
