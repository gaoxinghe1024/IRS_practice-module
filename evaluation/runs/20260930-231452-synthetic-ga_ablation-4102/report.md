# GA ablation (validation users) - synthetic data

Engine `icf`, 726 validation users (their test positive excluded from candidates). All methods in a row group use the same candidate pool and scores.

> Synthetic data: validates the optimiser implementation and trade-offs, not real quality.

| pool | objective | method | hits@10 | ild | novelty | fresh | fitness_full_objective | n |
|---|---|---|---|---|---|---|---|---|
| pool100 | full | ga_s0 | 0.0964 | 0.8787 | 0.3776 | 0.6582 | 0.7313 | 591 |
| pool100 | full | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.7035 | 591 |
| pool100 | full | mmr_0.7 | 0.1777 | 0.8713 | 0.3230 | 0.4584 | 0.7111 | 591 |
| pool20 | full | ga_s0 | 0.1540 | 0.8622 | 0.3465 | 0.5521 | 0.6217 | 591 |
| pool20 | full | ga_s1 | 0.1523 | 0.8622 | 0.3465 | 0.5524 | 0.6218 | 591 |
| pool20 | full | ga_s2 | 0.1557 | 0.8620 | 0.3465 | 0.5524 | 0.6218 | 591 |
| pool20 | full | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | full | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool20 | no_diversity | ga_s0 | 0.1472 | 0.8456 | 0.3461 | 0.5538 | 0.6192 | 591 |
| pool20 | no_diversity | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | no_diversity | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool20 | no_fresh | ga_s0 | 0.1607 | 0.8624 | 0.3481 | 0.4455 | 0.6160 | 591 |
| pool20 | no_fresh | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | no_fresh | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool20 | no_novelty | ga_s0 | 0.1827 | 0.8623 | 0.3246 | 0.5536 | 0.6197 | 591 |
| pool20 | no_novelty | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6089 | 591 |
| pool20 | no_novelty | mmr_0.7 | 0.1980 | 0.8565 | 0.3240 | 0.4496 | 0.6122 | 591 |
| pool50 | full | ga_s0 | 0.1201 | 0.8717 | 0.3635 | 0.6073 | 0.6931 | 591 |
| pool50 | full | greedy | 0.1929 | 0.8433 | 0.3234 | 0.4410 | 0.6716 | 591 |
| pool50 | full | mmr_0.7 | 0.1810 | 0.8653 | 0.3230 | 0.4548 | 0.6776 | 591 |

Exhaustive oracle on 12-candidate pools (K=5): 591 pools, mean fitness gap 0.0, max gap 0.0, optimal share 1.0.

MMR is a separately named baseline, not the GA. The oracle proves optimality only within each pool.
