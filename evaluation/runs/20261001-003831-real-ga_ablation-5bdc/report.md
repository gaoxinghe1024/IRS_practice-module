# GA ablation (validation users) - real data

Engine `icf`, 600 validation users (their test positive excluded from candidates). All methods in a row group use the same candidate pool and scores.

| pool | objective | method | hits@10 | ild | novelty | fresh | fitness_full_objective | n |
|---|---|---|---|---|---|---|---|---|
| pool100 | full | ga_s0 | 0.0267 | 0.8325 | 0.4974 | 0.3040 | 0.6455 | 599 |
| pool100 | full | greedy | 0.0384 | 0.7447 | 0.4384 | 0.1643 | 0.6162 | 599 |
| pool100 | full | mmr_0.7 | 0.0417 | 0.8027 | 0.4425 | 0.1785 | 0.6284 | 599 |
| pool20 | full | ga_s0 | 0.0301 | 0.8023 | 0.4724 | 0.2332 | 0.5575 | 599 |
| pool20 | full | ga_s1 | 0.0301 | 0.8021 | 0.4724 | 0.2337 | 0.5575 | 599 |
| pool20 | full | ga_s2 | 0.0301 | 0.8024 | 0.4724 | 0.2339 | 0.5575 | 599 |
| pool20 | full | greedy | 0.0384 | 0.7447 | 0.4384 | 0.1643 | 0.5394 | 599 |
| pool20 | full | mmr_0.7 | 0.0417 | 0.7801 | 0.4412 | 0.1745 | 0.5469 | 599 |
| pool20 | no_diversity | ga_s0 | 0.0284 | 0.7536 | 0.4718 | 0.2327 | 0.5493 | 599 |
| pool20 | no_diversity | greedy | 0.0384 | 0.7447 | 0.4384 | 0.1643 | 0.5394 | 599 |
| pool20 | no_diversity | mmr_0.7 | 0.0417 | 0.7801 | 0.4412 | 0.1745 | 0.5469 | 599 |
| pool20 | no_fresh | ga_s0 | 0.0334 | 0.8013 | 0.4757 | 0.1781 | 0.5546 | 599 |
| pool20 | no_fresh | greedy | 0.0384 | 0.7447 | 0.4384 | 0.1643 | 0.5394 | 599 |
| pool20 | no_fresh | mmr_0.7 | 0.0417 | 0.7801 | 0.4412 | 0.1745 | 0.5469 | 599 |
| pool20 | no_novelty | ga_s0 | 0.0401 | 0.8006 | 0.4414 | 0.2419 | 0.5543 | 599 |
| pool20 | no_novelty | greedy | 0.0384 | 0.7447 | 0.4384 | 0.1643 | 0.5394 | 599 |
| pool20 | no_novelty | mmr_0.7 | 0.0417 | 0.7801 | 0.4412 | 0.1745 | 0.5469 | 599 |
| pool50 | full | ga_s0 | 0.0317 | 0.8217 | 0.4885 | 0.2815 | 0.6153 | 599 |
| pool50 | full | greedy | 0.0384 | 0.7447 | 0.4384 | 0.1643 | 0.5903 | 599 |
| pool50 | full | mmr_0.7 | 0.0417 | 0.7922 | 0.4429 | 0.1781 | 0.6006 | 599 |

Exhaustive oracle on 12-candidate pools (K=5): 599 pools, mean fitness gap 3.286528305047963e-06, max gap 0.00196863045472373, optimal share 0.998330550918197.

MMR is a separately named baseline, not the GA. The oracle proves optimality only within each pool.
