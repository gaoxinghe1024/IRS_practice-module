# GA ablation (validation users) - synthetic data

Engine `icf`, 726 validation users (their test positive excluded from candidates). All methods in a row group use the same candidate pool and scores.

> Synthetic data: validates the optimiser implementation and trade-offs, not real quality.

| pool | objective | method | hits@10 | ild | novelty | fresh | fitness_full_objective | n |
|---|---|---|---|---|---|---|---|---|
| pool100 | full | ga_s0 | 0.1015 | 0.8779 | 0.3777 | 0.6587 | 0.7312 | 591 |
| pool100 | full | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.7035 | 591 |
| pool100 | full | mmr_0.7 | 0.1777 | 0.8713 | 0.3230 | 0.4584 | 0.7111 | 591 |
| pool20 | full | ga_s0 | 0.1540 | 0.8621 | 0.3465 | 0.5531 | 0.6218 | 591 |
| pool20 | full | ga_s1 | 0.1523 | 0.8622 | 0.3467 | 0.5531 | 0.6218 | 591 |
| pool20 | full | ga_s2 | 0.1540 | 0.8622 | 0.3465 | 0.5530 | 0.6218 | 591 |
| pool20 | full | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | full | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool20 | no_diversity | ga_s0 | 0.1472 | 0.8452 | 0.3464 | 0.5536 | 0.6191 | 591 |
| pool20 | no_diversity | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | no_diversity | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool20 | no_fresh | ga_s0 | 0.1607 | 0.8625 | 0.3483 | 0.4452 | 0.6160 | 591 |
| pool20 | no_fresh | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | no_fresh | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool20 | no_novelty | ga_s0 | 0.1861 | 0.8623 | 0.3244 | 0.5540 | 0.6196 | 591 |
| pool20 | no_novelty | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | no_novelty | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool50 | full | ga_s0 | 0.1269 | 0.8721 | 0.3632 | 0.6085 | 0.6931 | 591 |
| pool50 | full | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6716 | 591 |
| pool50 | full | mmr_0.7 | 0.1810 | 0.8653 | 0.3230 | 0.4548 | 0.6776 | 591 |

Exhaustive oracle on 12-candidate pools (K=5): 591 pools, mean fitness gap 3.47296662956934e-07, max gap 0.00020525232780754799, optimal share 0.9983079526226735.

MMR is a separately named baseline, not the GA. The oracle proves optimality only within each pool.
