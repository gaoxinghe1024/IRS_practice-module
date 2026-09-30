# Homepage experiment (algorithm 3) - synthetic data

Dataset version `synthetic-7a7d5d6a74`; ranking protocol (one held-out high-rated test title per user). Users evaluated: 594. Candidate universe: served, recommendable titles not in the user's train history (validation positive removed). Candidate pool for optimisers: top 20 of each engine; list length K=10; GA seeds: 3.

> Synthetic data: these numbers validate that the pipeline and algorithms run correctly on a generated world; they are not evidence of real recommendation quality.

## Main table (means; 95% bootstrap CIs in metrics.json)

| engine | optimizer | hits@5 | hits@10 | mrr@10 | ndcg@10 | ild | novelty | fresh | fitness | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| icf | no_rerank | 0.0960 | 0.1902 | 0.0548 | 0.0859 | 0.8431 | 0.3237 | 0.4274 | 0.6062 | 1.0000 | 594 |
| icf | greedy | 0.0960 | 0.1902 | 0.0548 | 0.0859 | 0.8431 | 0.3237 | 0.4274 | 0.6062 | 1.0000 | 594 |
| icf | ga | 0.0926 | 0.1397 | 0.0464 | 0.0682 | 0.8624 | 0.3468 | 0.5375 | 0.6191 | 1.0000 | 594 |
| mf | no_rerank | 0.1885 | 0.2778 | 0.1108 | 0.1498 | 0.8439 | 0.2712 | 0.4333 | 0.5725 | 0.9371 | 594 |
| mf | greedy | 0.1885 | 0.2778 | 0.1108 | 0.1498 | 0.8439 | 0.2712 | 0.4333 | 0.5725 | 0.9371 | 594 |
| mf | ga | 0.1835 | 0.2643 | 0.1082 | 0.1447 | 0.8613 | 0.2912 | 0.5465 | 0.5832 | 0.9371 | 594 |
| ncf | no_rerank | 0.1263 | 0.2407 | 0.0838 | 0.1195 | 0.8683 | 0.1848 | 0.4337 | 0.5756 | 0.2743 | 594 |
| ncf | greedy | 0.1263 | 0.2407 | 0.0838 | 0.1195 | 0.8683 | 0.1848 | 0.4337 | 0.5756 | 0.2743 | 594 |
| ncf | ga | 0.1246 | 0.2189 | 0.0814 | 0.1129 | 0.8798 | 0.1938 | 0.5125 | 0.5822 | 0.2857 | 594 |
| popularity | no_rerank | 0.2407 | 0.3619 | 0.1466 | 0.1967 | 0.8446 | 0.1015 | 0.3742 | 0.5328 | 0.4229 | 594 |
| popularity | greedy | 0.2407 | 0.3619 | 0.1466 | 0.1967 | 0.8446 | 0.1015 | 0.3742 | 0.5328 | 0.4229 | 594 |
| popularity | ga | 0.2407 | 0.3401 | 0.1435 | 0.1895 | 0.8591 | 0.1052 | 0.4702 | 0.5397 | 0.4229 | 594 |

`no_rerank` is the recall order truncated to K. When relevance is a monotone transform of the recall score, relevance-greedy selects the same list; the share of users where they coincide is reported per engine below (an alias, not an independent result).

### icf
- candidate recall@20/50/100: 0.3451 [0.3081, 0.3838] (n=594), 0.6566 [0.6162, 0.6919] (n=594), 0.8771 [0.8502, 0.9024] (n=594)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0505 [-0.0707, -0.0303]; mrr@10 -0.0084 [-0.0130, -0.0045]; ild +0.0193 [+0.0177, +0.0209]; novelty +0.0231 [+0.0210, +0.0256]; fresh +0.1101 [+0.1025, +0.1182]; fitness +0.0129 [+0.0120, +0.0139]
- GA hits@10 by seed: [0.13973, 0.1431, 0.1431]
- latency p50: recall 0.2 ms, GA 87.7 ms

### mf
- candidate recall@20/50/100: 0.4343 [0.3939, 0.4764] (n=594), 0.7020 [0.6666, 0.7408] (n=594), 0.8956 [0.8704, 0.9192] (n=594)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0135 [-0.0320, +0.0067]; mrr@10 -0.0026 [-0.0056, +0.0003]; ild +0.0175 [+0.0161, +0.0189]; novelty +0.0200 [+0.0181, +0.0219]; fresh +0.1131 [+0.1057, +0.1212]; fitness +0.0108 [+0.0102, +0.0115]
- GA hits@10 by seed: [0.26431, 0.26263, 0.26431]
- latency p50: recall 0.1 ms, GA 88.1 ms

### ncf
- candidate recall@20/50/100: 0.3586 [0.3216, 0.3973] (n=594), 0.6145 [0.5791, 0.6515] (n=594), 0.8704 [0.8434, 0.8956] (n=594)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0219 [-0.0387, -0.0067]; mrr@10 -0.0024 [-0.0043, -0.0006]; ild +0.0115 [+0.0103, +0.0128]; novelty +0.0090 [+0.0081, +0.0100]; fresh +0.0788 [+0.0727, +0.0855]; fitness +0.0066 [+0.0061, +0.0072]
- GA hits@10 by seed: [0.21886, 0.21717, 0.21717]
- latency p50: recall 0.4 ms, GA 81.0 ms

### popularity
- candidate recall@20/50/100: 0.5050 [0.4647, 0.5438] (n=594), 0.7374 [0.7054, 0.7694] (n=594), 0.9377 [0.9192, 0.9563] (n=594)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0219 [-0.0421, -0.0034]; mrr@10 -0.0031 [-0.0061, -0.0006]; ild +0.0145 [+0.0132, +0.0157]; novelty +0.0037 [+0.0034, +0.0040]; fresh +0.0960 [+0.0884, +0.1039]; fitness +0.0068 [+0.0063, +0.0073]
- GA hits@10 by seed: [0.34007, 0.34175, 0.33838]
- latency p50: recall 0.0 ms, GA 84.3 ms

## Validation-only tuning

ICF prediction damping (validation hits@10): {0.0: nan, 0.5: nan, 2.0: nan, 5.0: nan}

## Reading the results

- The GA optimises the list objective (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) inside the candidate pool; a higher fitness is not by itself evidence of better recommendations. Accuracy and diversity trade-offs are shown side by side.
- A title that is not in the candidate pool cannot be recovered by any optimiser (see candidate recall).
- MAE / RMSE are properties of the recall predictors and are reported by the explicit-rating experiment.

- note: ICF damping tuned on validation positives of 0 other users: {0.0: nan, 0.5: nan, 2.0: nan, 5.0: nan} -> 0.0
