# Homepage experiment (algorithm 3) - real data

Dataset version `real-89bfdea00b`; ranking protocol (one held-out high-rated test title per user). Users evaluated: 1500. Candidate universe: served, recommendable titles not in the user's train history (validation positive removed). Candidate pool for optimisers: top 20 of each engine; list length K=10; GA seeds: 3.

## Main table (means; 95% bootstrap CIs in metrics.json)

| engine | optimizer | hits@5 | hits@10 | mrr@10 | ndcg@10 | ild | novelty | fresh | fitness | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| icf | no_rerank | 0.0467 | 0.0620 | 0.0321 | 0.0392 | 0.7467 | 0.4439 | 0.1615 | 0.5382 | 0.1798 | 1500 |
| icf | greedy | 0.0467 | 0.0620 | 0.0321 | 0.0392 | 0.7467 | 0.4439 | 0.1615 | 0.5382 | 0.1798 | 1500 |
| icf | ga | 0.0427 | 0.0527 | 0.0303 | 0.0356 | 0.8022 | 0.4798 | 0.2335 | 0.5566 | 0.1767 | 1500 |
| mf | no_rerank | 0.0233 | 0.0413 | 0.0140 | 0.0203 | 0.7718 | 0.4264 | 0.1967 | 0.5408 | 0.1259 | 1500 |
| mf | greedy | 0.0233 | 0.0413 | 0.0140 | 0.0203 | 0.7718 | 0.4264 | 0.1967 | 0.5408 | 0.1259 | 1500 |
| mf | ga | 0.0213 | 0.0300 | 0.0124 | 0.0165 | 0.8229 | 0.4875 | 0.2624 | 0.5612 | 0.1245 | 1500 |
| ncf | no_rerank | 0.0193 | 0.0367 | 0.0120 | 0.0176 | 0.7503 | 0.1967 | 0.1802 | 0.5353 | 0.0093 | 1500 |
| ncf | greedy | 0.0193 | 0.0367 | 0.0120 | 0.0176 | 0.7503 | 0.1967 | 0.1802 | 0.5353 | 0.0093 | 1500 |
| ncf | ga | 0.0187 | 0.0313 | 0.0114 | 0.0160 | 0.8046 | 0.2122 | 0.2473 | 0.5490 | 0.0091 | 1500 |
| popularity | no_rerank | 0.0840 | 0.1367 | 0.0555 | 0.0742 | 0.7756 | 0.0424 | 0.3165 | 0.4604 | 0.0058 | 1500 |
| popularity | greedy | 0.0840 | 0.1367 | 0.0555 | 0.0742 | 0.7756 | 0.0424 | 0.3165 | 0.4604 | 0.0058 | 1500 |
| popularity | ga | 0.0833 | 0.1320 | 0.0550 | 0.0728 | 0.7973 | 0.0442 | 0.4285 | 0.4692 | 0.0060 | 1500 |

`no_rerank` is the recall order truncated to K. When relevance is a monotone transform of the recall score, relevance-greedy selects the same list; the share of users where they coincide is reported per engine below (an alias, not an independent result).

### icf
- candidate recall@20/50/100: 0.0867 [0.0720, 0.1007] (n=1500), 0.1440 [0.1260, 0.1613] (n=1500), 0.1987 [0.1787, 0.2180] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0093 [-0.0167, -0.0027]; mrr@10 -0.0018 [-0.0031, -0.0007]; ild +0.0555 [+0.0531, +0.0580]; novelty +0.0359 [+0.0339, +0.0382]; fresh +0.0720 [+0.0671, +0.0768]; fitness +0.0184 [+0.0177, +0.0192]
- GA hits@10 by seed: [0.05267, 0.05333, 0.05267]
- latency p50: recall 1.1 ms, GA 25.7 ms

### mf
- candidate recall@20/50/100: 0.0680 [0.0553, 0.0807] (n=1500), 0.1133 [0.0973, 0.1287] (n=1500), 0.1620 [0.1433, 0.1800] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0113 [-0.0180, -0.0053]; mrr@10 -0.0017 [-0.0027, -0.0008]; ild +0.0511 [+0.0487, +0.0537]; novelty +0.0611 [+0.0580, +0.0642]; fresh +0.0657 [+0.0612, +0.0709]; fitness +0.0204 [+0.0196, +0.0211]
- GA hits@10 by seed: [0.03, 0.02933, 0.03]
- latency p50: recall 1.6 ms, GA 26.1 ms

### ncf
- candidate recall@20/50/100: 0.0653 [0.0540, 0.0780] (n=1500), 0.1187 [0.1027, 0.1347] (n=1500), 0.1787 [0.1587, 0.1967] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0053 [-0.0107, +0.0000]; mrr@10 -0.0006 [-0.0013, -0.0000]; ild +0.0543 [+0.0525, +0.0562]; novelty +0.0155 [+0.0144, +0.0164]; fresh +0.0671 [+0.0628, +0.0720]; fitness +0.0137 [+0.0131, +0.0142]
- GA hits@10 by seed: [0.03133, 0.03067, 0.03133]
- latency p50: recall 3.6 ms, GA 25.0 ms

### popularity
- candidate recall@20/50/100: 0.2000 [0.1787, 0.2180] (n=1500), 0.3167 [0.2933, 0.3413] (n=1500), 0.4353 [0.4100, 0.4593] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0047 [-0.0120, +0.0027]; mrr@10 -0.0005 [-0.0016, +0.0005]; ild +0.0217 [+0.0205, +0.0228]; novelty +0.0018 [+0.0018, +0.0019]; fresh +0.1119 [+0.1073, +0.1167]; fitness +0.0088 [+0.0084, +0.0091]
- GA hits@10 by seed: [0.132, 0.132, 0.132]
- latency p50: recall 1.3 ms, GA 25.9 ms

## Validation-only tuning

ICF prediction damping (validation hits@10, full histories): {0.0: 0.03166666666666667, 0.5: 0.04833333333333333, 2.0: 0.058333333333333334, 5.0: 0.06333333333333334, 10.0: 0.065, 20.0: 0.06833333333333333, 50.0: 0.06833333333333333}

ICF user-mean prior weight k (validation hits@10 by history length; chosen by the mean):

| k | 5 | 8 | 15 | full | mean |
|---|---|---|---|---|---|
| 0.0000 | 0.0409 | 0.0305 | 0.0385 | 0.0683 | 0.0446 |
| 2.0000 | 0.0426 | 0.0305 | 0.0428 | 0.0683 | 0.0461 |
| 5.0000 | 0.0426 | 0.0286 | 0.0450 | 0.0683 | 0.0461 |
| 10.0000 | 0.0426 | 0.0286 | 0.0450 | 0.0667 | 0.0457 |
| 20.0000 | 0.0409 | 0.0286 | 0.0450 | 0.0683 | 0.0457 |

Served ICF parameters used for the test results above: {'damping': 20.0, 'user_mean_prior_weight': 5.0}.

Home content ranking: popularity weight g in (1 - g) * profile match + g * popular_quality (validation hits@10 by history length; chosen by the mean):

| g | 1 | 3 | 5 | mean |
|---|---|---|---|---|
| 0.0000 | 0.0089 | 0.0134 | 0.0053 | 0.0092 |
| 0.2500 | 0.0248 | 0.0269 | 0.0232 | 0.0250 |
| 0.5000 | 0.0319 | 0.0353 | 0.0285 | 0.0319 |
| 0.7500 | 0.0390 | 0.0370 | 0.0303 | 0.0354 |
| 0.9000 | 0.0426 | 0.0370 | 0.0446 | 0.0414 |

Served home parameters used below: {'content_popularity_weight': 0.9}.

## Short histories (warm-gate analysis, test positives)

Each user's train history is cut to a seeded random subset of n ratings (a new user who rated n titles); the rest of the history stays excluded from the candidates. `icf_exact` is the brief's formula (no damping, no prior); `content` is the home content ranking with the tuned popularity weight and `content_tags_only` the pure tag profile; `served_blend` mirrors the product: below the warm gate (8 ratings) a lambda = n / gate mix of ICF and the content profile, above it ICF. Hits@10, first 500 evaluation users.

| ratings | icf_served | icf_exact | content | content_tags_only | popularity | served_blend | users |
|---|---|---|---|---|---|---|---|
| 1 | 0.0160 | 0.0220 | 0.0560 | 0.0280 | 0.1380 | 0.0420 | 500 |
| 3 | 0.0280 | 0.0240 | 0.0540 | 0.0100 | 0.1380 | 0.0360 | 500 |
| 5 | 0.0340 | 0.0277 | 0.0425 | 0.0043 | 0.1319 | 0.0447 | 470 |
| 8 | 0.0299 | 0.0207 | 0.0483 | 0.0000 | 0.1287 | 0.0299 | 435 |
| 15 | 0.0285 | 0.0155 | 0.0337 | 0.0026 | 0.1192 | 0.0285 | 386 |
| 30 | 0.0279 | 0.0248 | 0.0373 | 0.0062 | 0.1211 | 0.0279 | 322 |

## Reading the results

- The GA optimises the list objective (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) inside the candidate pool; a higher fitness is not by itself evidence of better recommendations. Accuracy and diversity trade-offs are shown side by side.
- A title that is not in the candidate pool cannot be recovered by any optimiser (see candidate recall).
- MAE / RMSE are properties of the recall predictors and are reported by the explicit-rating experiment.

- note: ICF damping tuned on validation positives of 600 other users: {0.0: 0.03166666666666667, 0.5: 0.04833333333333333, 2.0: 0.058333333333333334, 5.0: 0.06333333333333334, 10.0: 0.065, 20.0: 0.06833333333333333, 50.0: 0.06833333333333333} -> 20.0
- note: ICF user-mean prior weight tuned on the same validation users (mean hits@10 over history lengths (5, 8, 15) and full): {0.0: {'5': 0.04085257548845471, '8': 0.030534351145038167, '15': 0.03854389721627409, 'full': 0.06833333333333333, 'mean': 0.04456603929577507}, 2.0: {'5': 0.04262877442273535, '8': 0.030534351145038167, '15': 0.042826552462526764, 'full': 0.06833333333333333, 'mean': 0.0460807528409084}, 5.0: {'5': 0.04262877442273535, '8': 0.02862595419847328, '15': 0.044967880085653104, 'full': 0.06833333333333333, 'mean': 0.04613898551004877}, 10.0: {'5': 0.04262877442273535, '8': 0.02862595419847328, '15': 0.044967880085653104, 'full': 0.06666666666666667, 'mean': 0.0457223188433821}, 20.0: {'5': 0.04085257548845471, '8': 0.02862595419847328, '15': 0.044967880085653104, 'full': 0.06833333333333333, 'mean': 0.04569493577647861}} -> 5.0
- note: Home content popularity weight tuned on validation (histories of (1, 3, 5) ratings): {0.0: {'1': 0.008865248226950355, '3': 0.013445378151260505, '5': 0.0053475935828877, 'mean': 0.00921940665369952}, 0.25: {'1': 0.024822695035460994, '3': 0.02689075630252101, '5': 0.023172905525846704, 'mean': 0.024962118954609567}, 0.5: {'1': 0.031914893617021274, '3': 0.03529411764705882, '5': 0.0285204991087344, 'mean': 0.031909836790938166}, 0.75: {'1': 0.03900709219858156, '3': 0.03697478991596639, '5': 0.030303030303030304, 'mean': 0.035428304139192755}, 0.9: {'1': 0.0425531914893617, '3': 0.03697478991596639, '5': 0.044563279857397504, 'mean': 0.041363753754241867}} -> 0.9
