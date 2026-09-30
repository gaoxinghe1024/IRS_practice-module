# Homepage experiment (algorithm 3) - synthetic data

Dataset version `synthetic-7a7d5d6a74`; ranking protocol (one held-out high-rated test title per user). Users evaluated: 476. Candidate universe: served, recommendable titles not in the user's train history (validation positive removed). Candidate pool for optimisers: top 20 of each engine; list length K=10; GA seeds: 3.

> Synthetic data: these numbers validate that the pipeline and algorithms run correctly on a generated world; they are not evidence of real recommendation quality.

## Main table (means; 95% bootstrap CIs in metrics.json)

| engine | optimizer | hits@5 | hits@10 | mrr@10 | ndcg@10 | ild | novelty | fresh | fitness | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| icf | no_rerank | 0.1660 | 0.2731 | 0.0987 | 0.1389 | 0.8457 | 0.3158 | 0.4252 | 0.5874 | 1.0000 | 476 |
| icf | greedy | 0.1660 | 0.2731 | 0.0987 | 0.1389 | 0.8457 | 0.3158 | 0.4252 | 0.5874 | 1.0000 | 476 |
| icf | ga | 0.1639 | 0.2290 | 0.0935 | 0.1254 | 0.8630 | 0.3396 | 0.5286 | 0.5982 | 0.9886 | 476 |
| mf | no_rerank | 0.1765 | 0.2731 | 0.1006 | 0.1407 | 0.8446 | 0.2719 | 0.4321 | 0.5723 | 0.9257 | 476 |
| mf | greedy | 0.1765 | 0.2731 | 0.1006 | 0.1407 | 0.8446 | 0.2719 | 0.4321 | 0.5723 | 0.9257 | 476 |
| mf | ga | 0.1723 | 0.2479 | 0.0970 | 0.1324 | 0.8618 | 0.2917 | 0.5456 | 0.5830 | 0.9257 | 476 |
| ncf | no_rerank | 0.1155 | 0.2374 | 0.0823 | 0.1172 | 0.8689 | 0.1839 | 0.4315 | 0.5756 | 0.2343 | 476 |
| ncf | greedy | 0.1155 | 0.2374 | 0.0823 | 0.1172 | 0.8689 | 0.1839 | 0.4315 | 0.5756 | 0.2343 | 476 |
| ncf | ga | 0.1155 | 0.2164 | 0.0801 | 0.1111 | 0.8804 | 0.1928 | 0.5092 | 0.5822 | 0.2457 | 476 |
| popularity | no_rerank | 0.2332 | 0.3613 | 0.1412 | 0.1923 | 0.8445 | 0.1010 | 0.3731 | 0.5319 | 0.3886 | 476 |
| popularity | greedy | 0.2332 | 0.3613 | 0.1412 | 0.1923 | 0.8445 | 0.1010 | 0.3731 | 0.5319 | 0.3886 | 476 |
| popularity | ga | 0.2353 | 0.3319 | 0.1377 | 0.1832 | 0.8594 | 0.1049 | 0.4698 | 0.5387 | 0.3886 | 476 |

`no_rerank` is the recall order truncated to K. When relevance is a monotone transform of the recall score, relevance-greedy selects the same list; the share of users where they coincide is reported per engine below (an alias, not an independent result).

### icf
- candidate recall@20/50/100: 0.4034 [0.3592, 0.4475] (n=476), 0.6723 [0.6323, 0.7164] (n=476), 0.8677 [0.8382, 0.8971] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0441 [-0.0651, -0.0273]; mrr@10 -0.0052 [-0.0079, -0.0028]; ild +0.0173 [+0.0157, +0.0191]; novelty +0.0237 [+0.0212, +0.0261]; fresh +0.1034 [+0.0956, +0.1120]; fitness +0.0108 [+0.0101, +0.0115]
- GA hits@10 by seed: [0.22899, 0.23319, 0.22899]
- latency p50: recall 0.2 ms, GA 29.5 ms

### mf
- candidate recall@20/50/100: 0.4307 [0.3887, 0.4748] (n=476), 0.6954 [0.6555, 0.7353] (n=476), 0.8950 [0.8677, 0.9223] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0252 [-0.0483, -0.0063]; mrr@10 -0.0035 [-0.0066, -0.0009]; ild +0.0173 [+0.0157, +0.0188]; novelty +0.0198 [+0.0177, +0.0219]; fresh +0.1134 [+0.1048, +0.1223]; fitness +0.0107 [+0.0099, +0.0114]
- GA hits@10 by seed: [0.2479, 0.2479, 0.2479]
- latency p50: recall 0.0 ms, GA 29.6 ms

### ncf
- candidate recall@20/50/100: 0.3508 [0.3046, 0.3950] (n=476), 0.6113 [0.5693, 0.6555] (n=476), 0.8592 [0.8277, 0.8887] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0210 [-0.0399, -0.0063]; mrr@10 -0.0021 [-0.0042, -0.0003]; ild +0.0115 [+0.0103, +0.0128]; novelty +0.0089 [+0.0079, +0.0100]; fresh +0.0777 [+0.0712, +0.0851]; fitness +0.0065 [+0.0060, +0.0072]
- GA hits@10 by seed: [0.21639, 0.21639, 0.21639]
- latency p50: recall 0.4 ms, GA 28.0 ms

### popularity
- candidate recall@20/50/100: 0.5042 [0.4601, 0.5504] (n=476), 0.7290 [0.6891, 0.7710] (n=476), 0.9328 [0.9096, 0.9538] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0294 [-0.0483, -0.0084]; mrr@10 -0.0035 [-0.0062, -0.0008]; ild +0.0149 [+0.0135, +0.0164]; novelty +0.0038 [+0.0035, +0.0042]; fresh +0.0966 [+0.0884, +0.1050]; fitness +0.0068 [+0.0063, +0.0073]
- GA hits@10 by seed: [0.33193, 0.33403, 0.33403]
- latency p50: recall 0.0 ms, GA 28.9 ms

## Validation-only tuning

ICF prediction damping (validation hits@10): {0.0: 0.15172413793103448, 0.5: 0.1793103448275862, 2.0: 0.20689655172413793, 5.0: 0.2206896551724138}

## Reading the results

- The GA optimises the list objective (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) inside the candidate pool; a higher fitness is not by itself evidence of better recommendations. Accuracy and diversity trade-offs are shown side by side.
- A title that is not in the candidate pool cannot be recovered by any optimiser (see candidate recall).
- MAE / RMSE are properties of the recall predictors and are reported by the explicit-rating experiment.

- note: ICF damping tuned on validation positives of 145 other users: {0.0: 0.15172413793103448, 0.5: 0.1793103448275862, 2.0: 0.20689655172413793, 5.0: 0.2206896551724138} -> 5.0
