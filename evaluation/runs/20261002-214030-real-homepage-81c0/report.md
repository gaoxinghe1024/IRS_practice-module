# Homepage experiment (algorithm 3) - real data

Dataset version `real-034ebd6a74`; ranking protocol (one held-out high-rated test title per user). Users evaluated: 1500. Candidate universe: served, recommendable titles not in the user's train history (validation positive removed). Candidate pool for optimisers: top 20 of each engine; list length K=10; GA seeds: 3.

## Main table (means; 95% bootstrap CIs in metrics.json)

| engine | optimizer | hits@5 | hits@10 | mrr@10 | ndcg@10 | ild | novelty | fresh | fitness | coverage | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| icf | no_rerank | 0.0467 | 0.0620 | 0.0319 | 0.0390 | 0.7636 | 0.4471 | 0.3585 | 0.5636 | 0.1652 | 1500 |
| icf | greedy | 0.0467 | 0.0620 | 0.0319 | 0.0390 | 0.7636 | 0.4471 | 0.3585 | 0.5636 | 0.1652 | 1500 |
| icf | ga | 0.0400 | 0.0467 | 0.0290 | 0.0333 | 0.8198 | 0.4818 | 0.4795 | 0.5849 | 0.1628 | 1500 |
| mf | no_rerank | 0.0220 | 0.0400 | 0.0137 | 0.0198 | 0.7814 | 0.4403 | 0.4043 | 0.5667 | 0.1167 | 1500 |
| mf | greedy | 0.0220 | 0.0400 | 0.0137 | 0.0198 | 0.7814 | 0.4403 | 0.4043 | 0.5667 | 0.1167 | 1500 |
| mf | ga | 0.0200 | 0.0247 | 0.0115 | 0.0147 | 0.8370 | 0.5007 | 0.5035 | 0.5899 | 0.1120 | 1500 |
| ncf | no_rerank | 0.0193 | 0.0367 | 0.0120 | 0.0176 | 0.7387 | 0.1967 | 0.5941 | 0.5732 | 0.0084 | 1500 |
| ncf | greedy | 0.0193 | 0.0367 | 0.0120 | 0.0176 | 0.7387 | 0.1967 | 0.5941 | 0.5732 | 0.0084 | 1500 |
| ncf | ga | 0.0180 | 0.0273 | 0.0106 | 0.0145 | 0.7852 | 0.2136 | 0.6789 | 0.5874 | 0.0084 | 1500 |
| popularity | no_rerank | 0.0840 | 0.1367 | 0.0555 | 0.0742 | 0.7922 | 0.0424 | 0.3681 | 0.4706 | 0.0052 | 1500 |
| popularity | greedy | 0.0840 | 0.1367 | 0.0555 | 0.0742 | 0.7922 | 0.0424 | 0.3681 | 0.4706 | 0.0052 | 1500 |
| popularity | ga | 0.0807 | 0.1287 | 0.0542 | 0.0714 | 0.8164 | 0.0440 | 0.4861 | 0.4812 | 0.0056 | 1500 |

`no_rerank` is the recall order truncated to K. When relevance is a monotone transform of the recall score, relevance-greedy selects the same list; the share of users where they coincide is reported per engine below (an alias, not an independent result).

### icf
- candidate recall@20/50/100: 0.0867 [0.0720, 0.1007] (n=1500), 0.1427 [0.1247, 0.1600] (n=1500), 0.1967 [0.1773, 0.2160] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0153 [-0.0227, -0.0080]; mrr@10 -0.0029 [-0.0044, -0.0015]; ild +0.0562 [+0.0536, +0.0588]; novelty +0.0347 [+0.0326, +0.0370]; fresh +0.1210 [+0.1154, +0.1263]; fitness +0.0212 [+0.0204, +0.0220]
- GA hits@10 by seed: [0.04667, 0.04667, 0.04667]
- latency p50: recall 1.3 ms, GA 30.2 ms

### mf
- candidate recall@20/50/100: 0.0640 [0.0527, 0.0760] (n=1500), 0.1100 [0.0940, 0.1240] (n=1500), 0.1580 [0.1393, 0.1767] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0153 [-0.0220, -0.0093]; mrr@10 -0.0022 [-0.0033, -0.0013]; ild +0.0555 [+0.0529, +0.0583]; novelty +0.0604 [+0.0573, +0.0635]; fresh +0.0991 [+0.0938, +0.1043]; fitness +0.0232 [+0.0224, +0.0240]
- GA hits@10 by seed: [0.02467, 0.02467, 0.02467]
- latency p50: recall 1.7 ms, GA 30.6 ms

### ncf
- candidate recall@20/50/100: 0.0653 [0.0540, 0.0780] (n=1500), 0.1187 [0.1027, 0.1347] (n=1500), 0.1787 [0.1587, 0.1967] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0093 [-0.0147, -0.0047]; mrr@10 -0.0014 [-0.0024, -0.0006]; ild +0.0465 [+0.0447, +0.0485]; novelty +0.0169 [+0.0158, +0.0180]; fresh +0.0848 [+0.0796, +0.0902]; fitness +0.0142 [+0.0135, +0.0147]
- GA hits@10 by seed: [0.02733, 0.02733, 0.02733]
- latency p50: recall 4.0 ms, GA 29.4 ms

### popularity
- candidate recall@20/50/100: 0.2000 [0.1787, 0.2180] (n=1500), 0.3167 [0.2933, 0.3413] (n=1500), 0.4353 [0.4100, 0.4593] (n=1500)
- no-rerank == greedy for 100.0% of users
- GA - greedy (paired bootstrap): hits@10 -0.0080 [-0.0160, +0.0007]; mrr@10 -0.0012 [-0.0024, +0.0000]; ild +0.0242 [+0.0230, +0.0253]; novelty +0.0017 [+0.0016, +0.0017]; fresh +0.1180 [+0.1133, +0.1229]; fitness +0.0106 [+0.0102, +0.0110]
- GA hits@10 by seed: [0.12867, 0.12933, 0.128]
- latency p50: recall 1.4 ms, GA 29.3 ms

## Validation-only tuning

ICF prediction damping (validation hits@10, full histories): {0.0: 0.03166666666666667, 0.5: 0.04833333333333333, 2.0: 0.058333333333333334, 5.0: 0.06333333333333334, 10.0: 0.065, 20.0: 0.06666666666666667, 50.0: 0.06666666666666667}

ICF user-mean prior weight k (validation hits@10 by history length; chosen by the mean):

| k | 5 | 8 | 15 | full | mean |
|---|---|---|---|---|---|
| 0.0000 | 0.0409 | 0.0305 | 0.0385 | 0.0667 | 0.0441 |
| 2.0000 | 0.0426 | 0.0305 | 0.0428 | 0.0667 | 0.0457 |
| 5.0000 | 0.0426 | 0.0286 | 0.0450 | 0.0667 | 0.0457 |
| 10.0000 | 0.0426 | 0.0286 | 0.0450 | 0.0667 | 0.0457 |
| 20.0000 | 0.0391 | 0.0286 | 0.0450 | 0.0683 | 0.0453 |

Served ICF parameters used for the test results above: {'damping': 20.0, 'user_mean_prior_weight': 5.0}.

Home content ranking: popularity weight g in (1 - g) * profile match + g * popular_quality (validation hits@10 by history length; chosen by the mean):

| g | 1 | 3 | 5 | mean |
|---|---|---|---|---|
| 0.0000 | 0.0071 | 0.0118 | 0.0107 | 0.0099 |
| 0.2500 | 0.0177 | 0.0252 | 0.0303 | 0.0244 |
| 0.5000 | 0.0337 | 0.0370 | 0.0428 | 0.0378 |
| 0.7500 | 0.0355 | 0.0319 | 0.0303 | 0.0326 |
| 0.9000 | 0.0426 | 0.0370 | 0.0428 | 0.0408 |

Served home parameters used below: {'content_popularity_weight': 0.9}.

## Short histories (warm-gate analysis, test positives)

Each user's train history is cut to a seeded random subset of n ratings (a new user who rated n titles); the rest of the history stays excluded from the candidates. `icf_exact` is the brief's formula (no damping, no prior); `content` is the home content ranking with the tuned popularity weight and `content_tags_only` the pure tag profile; `served_blend` mirrors the product: below the warm gate (8 ratings) a lambda = n / gate mix of ICF and the content profile, above it ICF. Hits@10, first 500 evaluation users.

| ratings | icf_served | icf_exact | content | content_tags_only | popularity | served_blend | users |
|---|---|---|---|---|---|---|---|
| 1 | 0.0160 | 0.0220 | 0.0620 | 0.0280 | 0.1380 | 0.0460 | 500 |
| 3 | 0.0280 | 0.0240 | 0.0540 | 0.0080 | 0.1380 | 0.0360 | 500 |
| 5 | 0.0319 | 0.0255 | 0.0447 | 0.0021 | 0.1319 | 0.0425 | 470 |
| 8 | 0.0299 | 0.0207 | 0.0414 | 0.0023 | 0.1287 | 0.0299 | 435 |
| 15 | 0.0285 | 0.0155 | 0.0363 | 0.0000 | 0.1192 | 0.0285 | 386 |
| 30 | 0.0279 | 0.0248 | 0.0311 | 0.0000 | 0.1211 | 0.0279 | 322 |

## Reading the results

- The GA optimises the list objective (0.4 relevance + 0.3 diversity + 0.2 novelty + 0.1 fresh) inside the candidate pool; a higher fitness is not by itself evidence of better recommendations. Accuracy and diversity trade-offs are shown side by side.
- A title that is not in the candidate pool cannot be recovered by any optimiser (see candidate recall).
- MAE / RMSE are properties of the recall predictors and are reported by the explicit-rating experiment.

- note: ICF damping tuned on validation positives of 600 other users: {0.0: 0.03166666666666667, 0.5: 0.04833333333333333, 2.0: 0.058333333333333334, 5.0: 0.06333333333333334, 10.0: 0.065, 20.0: 0.06666666666666667, 50.0: 0.06666666666666667} -> 20.0
- note: ICF user-mean prior weight tuned on the same validation users (mean hits@10 over history lengths (5, 8, 15) and full): {0.0: {'5': 0.04085257548845471, '8': 0.030534351145038167, '15': 0.03854389721627409, 'full': 0.06666666666666667, 'mean': 0.04414937262910841}, 2.0: {'5': 0.04262877442273535, '8': 0.030534351145038167, '15': 0.042826552462526764, 'full': 0.06666666666666667, 'mean': 0.04566408617424174}, 5.0: {'5': 0.04262877442273535, '8': 0.02862595419847328, '15': 0.044967880085653104, 'full': 0.06666666666666667, 'mean': 0.0457223188433821}, 10.0: {'5': 0.04262877442273535, '8': 0.02862595419847328, '15': 0.044967880085653104, 'full': 0.06666666666666667, 'mean': 0.0457223188433821}, 20.0: {'5': 0.03907637655417407, '8': 0.02862595419847328, '15': 0.044967880085653104, 'full': 0.06833333333333333, 'mean': 0.045250886042908445}} -> 5.0
- note: Home content popularity weight tuned on validation (histories of (1, 3, 5) ratings): {0.0: {'1': 0.0070921985815602835, '3': 0.011764705882352941, '5': 0.0106951871657754, 'mean': 0.009850697209896208}, 0.25: {'1': 0.01773049645390071, '3': 0.025210084033613446, '5': 0.030303030303030304, 'mean': 0.024414536930181488}, 0.5: {'1': 0.03368794326241135, '3': 0.03697478991596639, '5': 0.0427807486631016, 'mean': 0.03781449394715978}, 0.75: {'1': 0.03546099290780142, '3': 0.031932773109243695, '5': 0.030303030303030304, 'mean': 0.03256559877335847}, 0.9: {'1': 0.0425531914893617, '3': 0.03697478991596639, '5': 0.0427807486631016, 'mean': 0.04076957668947656}} -> 0.9
