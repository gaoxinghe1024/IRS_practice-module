# Human evaluation protocol

The real-data evaluation in `evaluation/runs/` covers what the Kaggle snapshot supports: ranking and rating
metrics, constraint compliance, robustness and latency. Three questions need human judgement that the snapshot
cannot supply. This protocol describes how to collect those judgements so the existing experiments pick them up
unchanged.

| Question | File to produce | Picked up by |
|---|---|---|
| Are the "similar titles" (algorithm 1) actually similar? | `similarity_labels.jsonl` | `evaluate --experiment similar`; also trains scorer B |
| Does chat find the right titles for real requests? | `rag_queries.jsonl` | `evaluate --experiment chat` (labelled set) |
| Can people complete the main tasks? | `system_tasks.jsonl` | reported next to the system evaluation |

Formats: `docs/data-contracts.md` §5; one example row each in `templates/`.

## 1. Similarity labels

1. **Query sample.** Draw 120 query titles with a fixed seed from the recommendable pool, stratified by popularity
   (40 popular, 40 mid, 40 long-tail). Write the seed and the list down before labelling starts.
2. **Pooling.** For each query, pool the top 10 of every scorer (weighted attributes, embedding, hybrid; logistic
   only if trained). De-duplicate the pool and shuffle it, and hide which scorer proposed each title.
3. **Judging.** Two annotators judge each pair independently on a 0-3 scale: 0 not similar, 1 somewhat
   (shares a few traits), 2 similar (same kind of show), 3 very similar (would recommend to the same viewer
   first). Annotators may read the synopsis and tags on the detail page. Popularity and score are not part of
   similarity.
4. **Agreement.** Report weighted Cohen's kappa on the raw labels. Disagreements of 2 or more points are
   discussed, and the resolved label is recorded with `note`.
5. **File.** Write one row per query: `positives` = pooled items with a final grade >= 1 (with `grade`), and
   `label_source: "human_annotation"`. Items judged 0 are simply absent; they stay "unlabelled", not "proven
   dissimilar".
6. **Leakage.** Franchise members are grouped automatically (relations) when the file is split into
   train / validation / test for scorer B. Never tune weights on the test part.

## 2. Chat queries

1. Collect 100+ requests written by people who did not build the system. Examples: "a short, relaxing slice of
   life, no romance", "something like Steins;Gate but shorter", "what is the sequel to X?".
2. For each request record `intent`, any `hard_constraints` it states, and `relevant_item_keys`. A title is
   relevant if an annotator would accept it as a good answer *and* it satisfies the stated hard constraints.
3. Also include requests the system should decline: questions about scenes or episodes without sources, and
   out-of-scope requests. Mark these `answerable: false`.
4. Keep 20 % of the queries aside as a final test set, and look at it only once.

## 3. Task success sessions

For each participant (5-8 people): give the tasks from the user guide (find a title, build a collection and get
recommendations, chat with a numeric constraint, make and export a 15-hour plan). Record `steps`, `success`,
time and any help given. Do not coach during the task.

## 4. Reporting

Human-judged results are reported separately from the automatic metrics, with the number of queries,
annotators, agreement and the date. Until these files exist, the corresponding capabilities stay marked
`human_evaluation_pending` in `docs/delivery-status.md`.
