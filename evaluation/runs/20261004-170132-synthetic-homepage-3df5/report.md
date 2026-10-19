# Homepage experiment (algorithm 3) - synthetic data

Dataset version `synthetic-7a7d5d6a74`; ranking protocol (one held-out high-rated test title per user). Users evaluated: 476. Candidate universe: served, recommendable titles not in the user's train history (validation positive removed). Candidate pool for optimisers: top 20 of each engine; list length K=10; GA seeds: 3.

> Synthetic data: these numbers validate that the pipeline and algorithms run correctly on a generated world; they are not evidence of real recommendation quality.

## Main table (means; 95% bootstrap CIs in metrics.json)

| engine | optimizer | hits@5 | hits@10 | mrr@10 | ndcg@10 | ild | novelty | fresh | fitness | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| icf | no_rerank | 0.1744 | 0.2626 | 0.0992 | 0.1371 | 0.8455 | 0.3166 | 0.4256 | 0.5881 | 1.0000 | 476 |
| icf | greedy | 0.1744 | 0.2626 | 0.0992 | 0.1371 | 0.8455 | 0.3166 | 0.4256 | 0.5881 | 1.0000 | 476 |
| icf | ga | 0.1723 | 0.2332 | 0.0958 | 0.1281 | 0.8625 | 0.3410 | 0.5315 | 0.5989 | 0.9943 | 476 |
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
- candidate recall@20/50/100: 0.4118 [0.3676, 0.4559] (n=476), 0.6660 [0.6240, 0.7080] (n=476), 0.8677 [0.8382, 0.8971] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0294 [-0.0483, -0.0126]; mrr@10 -0.0035 [-0.0062, -0.0011]; ild +0.0171 [+0.0153, +0.0188]; novelty +0.0244 [+0.0221, +0.0267]; fresh +0.1059 [+0.0971, +0.1145]; fitness +0.0108 [+0.0100, +0.0115]
- GA hits@10 by seed: [0.23319, 0.22899, 0.23109]
- latency p50: recall 0.2 ms, GA 26.3 ms

### mf
- candidate recall@20/50/100: 0.4307 [0.3887, 0.4748] (n=476), 0.6954 [0.6555, 0.7353] (n=476), 0.8950 [0.8677, 0.9223] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0252 [-0.0483, -0.0063]; mrr@10 -0.0035 [-0.0066, -0.0009]; ild +0.0173 [+0.0157, +0.0188]; novelty +0.0198 [+0.0177, +0.0219]; fresh +0.1134 [+0.1048, +0.1223]; fitness +0.0107 [+0.0099, +0.0114]
- GA hits@10 by seed: [0.2479, 0.2479, 0.2479]
- latency p50: recall 0.0 ms, GA 26.3 ms

### ncf
- candidate recall@20/50/100: 0.3508 [0.3046, 0.3950] (n=476), 0.6113 [0.5693, 0.6555] (n=476), 0.8592 [0.8277, 0.8887] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0210 [-0.0399, -0.0063]; mrr@10 -0.0021 [-0.0042, -0.0003]; ild +0.0115 [+0.0103, +0.0128]; novelty +0.0089 [+0.0079, +0.0100]; fresh +0.0777 [+0.0712, +0.0851]; fitness +0.0065 [+0.0060, +0.0072]
- GA hits@10 by seed: [0.21639, 0.21639, 0.21639]
- latency p50: recall 0.4 ms, GA 25.0 ms

### popularity
- candidate recall@20/50/100: 0.5042 [0.4601, 0.5504] (n=476), 0.7290 [0.6891, 0.7710] (n=476), 0.9328 [0.9096, 0.9538] (n=476)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0294 [-0.0483, -0.0084]; mrr@10 -0.0035 [-0.0062, -0.0008]; ild +0.0149 [+0.0135, +0.0164]; novelty +0.0038 [+0.0035, +0.0042]; fresh +0.0966 [+0.0884, +0.1050]; fitness +0.0068 [+0.0063, +0.0073]
- GA hits@10 by seed: [0.33193, 0.33403, 0.33403]
- latency p50: recall 0.0 ms, GA 25.9 ms

## Validation-only tuning

ICF prediction damping (validation hits@10, full histories): {0.0: 0.15172413793103448, 0.5: 0.1793103448275862, 2.0: 0.20689655172413793, 5.0: 0.2206896551724138, 10.0: 0.21379310344827587, 20.0: 0.2206896551724138, 50.0: 0.2206896551724138}

ICF user-mean prior weight k (validation hits@10 by history length; chosen by the mean):

| k | 5 | 8 | 15 | full | mean |
|---|---|---|---|---|---|
| 0.0000 | 0.1357 | 0.2393 | 0.1667 | 0.2207 | 0.1906 |
| 2.0000 | 0.1357 | 0.2393 | 0.1667 | 0.2138 | 0.1889 |
| 5.0000 | 0.1357 | 0.2393 | 0.1778 | 0.2207 | 0.1934 |
| 10.0000 | 0.1500 | 0.2393 | 0.1556 | 0.2345 | 0.1948 |
| 20.0000 | 0.1357 | 0.2308 | 0.1667 | 0.2207 | 0.1885 |

Served ICF parameters used for the test results above: {'damping': 5.0, 'user_mean_prior_weight': 10.0}.

Home content ranking: popularity weight g in (1 - g) * profile match + g * popular_quality (validation hits@10 by history length; chosen by the mean):

| g | 1 | 3 | 5 | mean |
|---|---|---|---|---|
| 0.0000 | 0.0530 | 0.0759 | 0.1143 | 0.0811 |
| 0.2500 | 0.1288 | 0.1862 | 0.2143 | 0.1764 |
| 0.5000 | 0.2424 | 0.2897 | 0.2929 | 0.2750 |
| 0.7500 | 0.2803 | 0.2966 | 0.3000 | 0.2923 |
| 0.9000 | 0.2576 | 0.2966 | 0.3000 | 0.2847 |

Served home parameters used below: {'content_popularity_weight': 0.75}.

## Short histories (warm-gate analysis, test positives)

Each user's train history is cut to a seeded random subset of n ratings (a new user who rated n titles); the rest of the history stays excluded from the candidates. `icf_exact` is the brief's formula (no damping, no prior); `content` is the home content ranking with the tuned popularity weight and `content_tags_only` the pure tag profile; `served_blend` mirrors the product: below the warm gate (8 ratings) a lambda = n / gate mix of ICF and the content profile, above it ICF. Hits@10, first 500 evaluation users.

| ratings | icf_served | icf_exact | content | content_tags_only | popularity | served_blend | users |
|---|---|---|---|---|---|---|---|
| 1 | 0.0898 | 0.0801 | 0.3277 | 0.0898 | 0.3714 | 0.3155 | 412 |
| 3 | 0.1238 | 0.1165 | 0.3495 | 0.1044 | 0.3714 | 0.1918 | 412 |
| 5 | 0.1603 | 0.1349 | 0.3359 | 0.1145 | 0.3715 | 0.2545 | 393 |
| 8 | 0.1961 | 0.1148 | 0.3389 | 0.1260 | 0.3725 | 0.1961 | 357 |
| 15 | 0.2082 | 0.1388 | 0.3347 | 0.1429 | 0.3714 | 0.2082 | 245 |
| 30 | 0.3390 | 0.3136 | 0.3559 | 0.1610 | 0.3644 | 0.3390 | 118 |

## Reading the results

- The GA optimises the list objective (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) inside the candidate pool; a higher fitness is not by itself evidence of better recommendations. Accuracy and diversity trade-offs are shown side by side.
- A title that is not in the candidate pool cannot be recovered by any optimiser (see candidate recall).
- MAE / RMSE are properties of the recall predictors and are reported by the explicit-rating experiment.

- note: ICF damping tuned on validation positives of 145 other users: {0.0: 0.15172413793103448, 0.5: 0.1793103448275862, 2.0: 0.20689655172413793, 5.0: 0.2206896551724138, 10.0: 0.21379310344827587, 20.0: 0.2206896551724138, 50.0: 0.2206896551724138} -> 5.0
- note: ICF user-mean prior weight tuned on the same validation users (mean hits@10 over history lengths (5, 8, 15) and full): {0.0: {'5': 0.1357142857142857, '8': 0.23931623931623933, '15': 0.16666666666666666, 'full': 0.2206896551724138, 'mean': 0.19059671171740136}, 2.0: {'5': 0.1357142857142857, '8': 0.23931623931623933, '15': 0.16666666666666666, 'full': 0.21379310344827587, 'mean': 0.18887257378636688}, 5.0: {'5': 0.1357142857142857, '8': 0.23931623931623933, '15': 0.17777777777777778, 'full': 0.2206896551724138, 'mean': 0.19337448949517916}, 10.0: {'5': 0.15, '8': 0.23931623931623933, '15': 0.15555555555555556, 'full': 0.23448275862068965, 'mean': 0.19483863837312115}, 20.0: {'5': 0.1357142857142857, '8': 0.23076923076923078, '15': 0.16666666666666666, 'full': 0.2206896551724138, 'mean': 0.18845995958064923}} -> 10.0
- note: Home content popularity weight tuned on validation (histories of (1, 3, 5) ratings): {0.0: {'1': 0.05303030303030303, '3': 0.07586206896551724, '5': 0.11428571428571428, 'mean': 0.08105936209384484}, 0.25: {'1': 0.12878787878787878, '3': 0.18620689655172415, '5': 0.21428571428571427, 'mean': 0.17642682987510575}, 0.5: {'1': 0.24242424242424243, '3': 0.2896551724137931, '5': 0.29285714285714287, 'mean': 0.27497885256505944}, 0.75: {'1': 0.2803030303030303, '3': 0.296551724137931, '5': 0.3, 'mean': 0.2922849181469871}, 0.9: {'1': 0.25757575757575757, '3': 0.296551724137931, '5': 0.3, 'mean': 0.2847091605712295}} -> 0.75
