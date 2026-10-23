# GA ablation (validation users) - synthetic data

Engine `icf`, 726 validation users (their test positive excluded from candidates). All methods in a row group use the same candidate pool and scores.

> Synthetic data: validates the optimiser implementation and trade-offs, not real quality.

| pool | objective | method | hits@10 | ild | novelty | fresh | fitness_full_objective | n |
|---|---|---|---|---|---|---|---|---|
| pool100 | full | ga_s0 | 0.2098 | 0.8751 | 0.3579 | 0.6342 | 0.7009 | 591 |
| pool100 | full | greedy | 0.2792 | 0.8447 | 0.3143 | 0.4443 | 0.6805 | 591 |
| pool100 | full | mmr_0.7 | 0.2673 | 0.8645 | 0.3167 | 0.4562 | 0.6852 | 591 |
| pool20 | full | ga_s0 | 0.2487 | 0.8627 | 0.3378 | 0.5520 | 0.6016 | 591 |
| pool20 | full | ga_s1 | 0.2470 | 0.8629 | 0.3382 | 0.5514 | 0.6016 | 591 |
| pool20 | full | ga_s2 | 0.2487 | 0.8630 | 0.3382 | 0.5508 | 0.6016 | 591 |
| pool20 | full | greedy | 0.2792 | 0.8447 | 0.3143 | 0.4443 | 0.5908 | 591 |
| pool20 | full | mmr_0.7 | 0.2741 | 0.8551 | 0.3154 | 0.4504 | 0.5930 | 591 |
| pool20 | no_diversity | ga_s0 | 0.2504 | 0.8467 | 0.3369 | 0.5513 | 0.5991 | 591 |
| pool20 | no_diversity | greedy | 0.2792 | 0.8447 | 0.3143 | 0.4443 | 0.5908 | 591 |
| pool20 | no_diversity | mmr_0.7 | 0.2741 | 0.8551 | 0.3154 | 0.4504 | 0.5930 | 591 |
| pool20 | no_fresh | ga_s0 | 0.2437 | 0.8619 | 0.3382 | 0.4486 | 0.5963 | 591 |
| pool20 | no_fresh | greedy | 0.2792 | 0.8447 | 0.3143 | 0.4443 | 0.5908 | 591 |
| pool20 | no_fresh | mmr_0.7 | 0.2741 | 0.8551 | 0.3154 | 0.4504 | 0.5930 | 591 |
| pool20 | no_novelty | ga_s0 | 0.2758 | 0.8627 | 0.3165 | 0.5520 | 0.5996 | 591 |
| pool20 | no_novelty | greedy | 0.2792 | 0.8447 | 0.3143 | 0.4443 | 0.5908 | 591 |
| pool20 | no_novelty | mmr_0.7 | 0.2741 | 0.8551 | 0.3154 | 0.4504 | 0.5930 | 591 |
| pool50 | full | ga_s0 | 0.2284 | 0.8700 | 0.3493 | 0.5969 | 0.6648 | 591 |
| pool50 | full | greedy | 0.2792 | 0.8447 | 0.3143 | 0.4443 | 0.6488 | 591 |
| pool50 | full | mmr_0.7 | 0.2741 | 0.8597 | 0.3158 | 0.4535 | 0.6523 | 591 |

Exhaustive oracle on 12-candidate pools (K=5): 591 pools, mean fitness gap 2.2992353432194607e-06, max gap 0.0010913291530414426, optimal share 0.9966159052453468.

MMR is a separately named baseline, not the GA. The oracle proves optimality only within each pool.
