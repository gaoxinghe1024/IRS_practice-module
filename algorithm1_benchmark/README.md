# Algorithm 1 synthetic benchmark

This directory contains the reproducible Algorithm 1 comparison on 1,000
synthetic query targets. The four methods share the same five catalog fields:
genre, theme, demographic, studio, and episode count.

The labels are generated before ranking from the separately specified metadata
oracle in `generate_synthetic_1000.py`. They are not scraped MAL
recommendations and do not estimate real-user accuracy. The oracle uses seeded
noise, fixed easy / medium / hard pools, and a different weighting from method
A. Its parameters were adjusted after an initial pilot run, so the final score
is not an independent pre-registered estimate. No target is selected after
observing final predictions (`posthoc_selection=false`). Synthetic targets may
overlap weak-supervision training endpoints (949 of 1,000 targets in the
published run); this is a regression benchmark, not a leakage-free holdout.

Run from this directory with the repository virtual environment:

```text
python generate_synthetic_1000.py --output results/synthetic_1000
```

The published outputs are `results/synthetic_1000/testset.jsonl`,
`predictions.jsonl`, `per_query_metrics.jsonl`, `metrics.csv`, `summary.json`,
and `testset_manifest.json`. `coverage_at_5` is the recall-style coverage
metric `|Top5 ∩ truth| / |truth|`; `hit_rate_at_5` is the binary metric that
only asks whether at least one recommendation was hit. Both are reported to
avoid confusing the two meanings of “Top-5 hit rate”.

The benchmark is intended for deterministic regression and controlled method
comparison. Independent expert or user labels are required before making a
deployment claim.
