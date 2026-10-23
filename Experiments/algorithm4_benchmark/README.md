# Algorithm 4: chat to find anime (LLM + KB + RAG)

Offline, reproducible evaluation of the chat page of AnimeFinder v1.4.1
(`backend/services/rag.py`, git blob 3b5f6dc, commit 426db20 of the team
repository). The retrieval pipeline is frozen in `algo/rag_snapshot.py` together
with its seed knowledge base of 24 titles and 12 scenes. The AniLogic chat in
`app/chat` is not covered by this run: the method (R/N/T conditions, scoring,
bootstrap CIs) carries over, the numbers do not.

## What was tested

Does RAG improve the chat-to-find-anime page, which of six LLMs should it use,
and is the template pipeline still worth keeping? 80 queries (74 Chinese,
6 English) over six question types (scene 24, constraint 16, trap 12, similar 10,
mood 10, factoid 8) were run under three conditions:

- **R**: the production retrieval (lexical scoring, scene-intent threshold 0.34)
  puts knowledge-base text into the prompt.
- **N**: the same models with a role line only, no retrieval.
- **T**: the production template pipeline, with no LLM.

Models: Hy3, Deepseek-V4.1-Flash, GLM-5.3, GLM-5.3-Flash, MiniMax-M3, Kimi-K3.
That is 160 calls per model (80 questions x R/N), 80 template outputs and 240
judge calls. Three LLM judges score all answers listwise; the same-vendor judge is
excluded and each judge's scores are mean-centered. Metric definitions are in
`REPORT.md`, Appendix B.

## Results (R group, centered judge scores)

| Model               | Judge (raw) | Centered (rank) | >20 s | p95    | Output tokens/call |
| ------------------- | ----------- | --------------- | ----- | ------ | ------------------ |
| GLM-5.3-Flash       | 4.08        | 0.45 (1)        | 6/80  | 20.5 s | 313                |
| GLM-5.3             | 4.07        | 0.44 (2)        | 2/80  | 15.6 s | 305                |
| Kimi-K3             | 4.12        | 0.43 (3)        | 3/80  | 19.7 s | 277                |
| MiniMax-M3          | 4.02        | 0.33 (4)        | 12/80 | 32.4 s | 528                |
| Deepseek-V4.1-Flash | 3.99        | 0.30 (5)        | 0/80  | 8.1 s  | 628                |
| Hy3                 | 4.03        | 0.28 (6)        | 4/80  | 19.1 s | 790                |

RAG lift: judge +0.41 to +0.67, episode accuracy +0.36 to +0.64, in-KB citation
(reply level) +0.59 to +0.80. The only significant regression is factoid
(-0.50 to -0.62), caused by the named work missing from the prompt text.

## Findings

1. Keep RAG on the chat page. The gains are broad for about +400 net input
   tokens per query. Until named-work recall is fixed, do not send factoid
   questions through RAG (REPORT Recommendations 1 and 3).
2. The model choice depends on the scenario. The centered ranking has two tiers:
   GLM-5.3-Flash, GLM-5.3 and Kimi-K3 are significantly above MiniMax-M3,
   Deepseek-V4.1-Flash and Hy3 (9 of 15 pairwise gaps, all across the tiers).
   For quality use GLM-5.3-Flash or GLM-5.3, for latency Deepseek, for stable
   quality with the fewest output tokens Kimi-K3. Hy3 ties with the bottom tier
   and has the most output tokens per call (790). See Recommendation 2.
3. Fix named-work recall. 18 of the 80 queries name a KB work (8 factoid,
   10 similar). The work is missing from the R prompt for 11 of them (factoid 5,
   similar 6): 10 are pushed out of the top 5 by mention-demotion, which is
   intended for similar queries and a defect for factoid, and 1 is removed by
   the constraint filter. It is present for the other 7: 4 with its own entry
   row and 3 only through famous-scene lines
   (`results/extra/named_recall_per_question.csv`).
4. Keep the template pipeline only as an offline fallback. It cannot refuse, so
   it needs a no-match branch.

The full reasoning is in `REPORT.md`.

## Caveats

- Single run, a knowledge base of famous titles (ceiling effects), LLM judges
  without human ground truth. Judge centering changes the ranking, and the length
  and faithfulness dimensions favor R.
- Two scene entries (S08 and S12, two phrasings each) are suspected to differ from
  the real animations. Excluding all four questions the lift stays significant:
  +0.426 [+0.241, +0.611], cluster bootstrap over questions
  (`results/extra/scene_sensitivity_cluster.csv`).
- All answers came through the CodeBuddy CLI (2.158.0, headless, single turn), not
  through vendor API keys, so absolute scores, latencies and timeout rates are not
  end-to-end numbers for the deployed system. The system code was run as a Python
  module, not as the service. See "Scope of what was executed" in `REPORT.md`.
- No vendor API was called from this folder. The live backends were only exercised
  against a fake OpenAI-compatible server on 127.0.0.1 (`tests/test_backends.py`).

## Reproduce

```bash
pip install -r requirements.txt
python -m unittest discover -s tests
python harness/analyze.py
```

Replay mode needs no credentials or network and uses the standard library only
(tested with Python 3.10 on Windows 11). matplotlib 3.8.0 is needed for the
figures. Live generation is opt-in:

```bash
LLM_API_BASE=<OpenAI-compatible base URL> LLM_API_KEY=<key> \
  python harness/run_generation.py --group R --model <model> --backend openai
```

`--model` takes a display name from `--list-models` or a raw model id.
`LLM_MODEL` is the fallback id when a call does not name a model.

| Output | Command |
| --- | --- |
| Summary tables in `results/replay/` | `python harness/analyze.py` (replays `results/scored.csv`) |
| Rule metrics re-scored from the raw replies | `python harness/analyze.py --rescore` |
| `results/scored.csv` | `python harness/analyze.py --rescore --write-scored` |
| Judge aggregation, pairwise centered matrix | `python harness/judge.py --aggregate`, `python harness/analyze.py --pairwise-matrix` |
| Tables 1-2 and the two derived CSVs in `results/extra/` | `python harness/analyze.py --tables` (95% CI: percentile bootstrap over questions, 10,000 resamples) |
| Figures in `results/figures/` | `python tools/make_figures.py` (needs matplotlib) |

Tables 3-8 and the other files in `results/extra/` come from the team's offline
analysis scripts and are stored as they are.

## Layout

```
algo/      rag_snapshot.py (frozen pipeline), prompts.py, llm_backend.py
data/      testset_v1.jsonl, aliases.json, kb/anime.db, kb/anime_kb_export.json
harness/   run_generation.py, score.py, judge.py, analyze.py, models.yaml
tools/     verify_snapshot.py, make_figures.py
tests/     unittest suite (parity, scorers, replay, tables, figures, backends, docs)
results/   raw and judge outputs, retrieval cache, scored.csv, tables/, figures/, extra/
```

## Notes

- `tools/verify_snapshot.py <path to rag.py>` compares every function of the
  snapshot with the system file (comments and whitespace ignored); all 12 are
  identical. The system file is available with
  `git show 426db20:backend/services/rag.py`.
- `data/kb/anime.db` holds the `anime` (24 rows) and `scenes` (12 rows) tables of
  the AnimeFinder v1.4.1 demo seed (`ANIME_SEED` and `SCENE_SEED` in
  `backend/database.py`), rows unchanged. `mal_id` is 0 and `image_url` is empty
  for every row, and the file holds no credentials.
  `data/kb/anime_kb_export.json` is a readable export of the same content.
- `<profile>`, `<workspace>` and `<pkg>` in the saved replies replace local
  directory names.
- The test set is Chinese-first because the knowledge base and alias tables carry
  Chinese names and famous-scene descriptions; the 6 English questions are
  analyzed separately.
- Main experiment 2026-09-29, R+ field-completion follow-up 2026-09-30.
- `CodeBuddyCliBackend` needs a WorkBuddy/CodeBuddy login and only documents how
  the saved outputs were produced.
- Reuse terms for the knowledge base and for the frozen `algo/rag_snapshot.py`
  are not stated here; the repository owner sets them.
