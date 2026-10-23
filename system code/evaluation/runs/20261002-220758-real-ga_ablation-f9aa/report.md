# GA ablation (validation users) - real data

Engine `icf`, 600 validation users (their test positive excluded from candidates). All methods in a row group use the same candidate pool and scores.

| pool | objective | method | hits@10 | ild | novelty | fresh | fitness_full_objective | n |
|---|---|---|---|---|---|---|---|---|
| pool100 | full | ga_s0 | 0.0217 | 0.8503 | 0.5011 | 0.5643 | 0.6746 | 599 |
| pool100 | full | greedy | 0.0384 | 0.7616 | 0.4415 | 0.3618 | 0.6414 | 599 |
| pool100 | full | mmr_0.7 | 0.0401 | 0.8210 | 0.4465 | 0.3785 | 0.6542 | 599 |
| pool20 | full | ga_s0 | 0.0284 | 0.8188 | 0.4760 | 0.4765 | 0.5853 | 599 |
| pool20 | full | ga_s1 | 0.0284 | 0.8186 | 0.4760 | 0.4763 | 0.5853 | 599 |
| pool20 | full | ga_s2 | 0.0284 | 0.8188 | 0.4760 | 0.4760 | 0.5853 | 599 |
| pool20 | full | greedy | 0.0384 | 0.7616 | 0.4415 | 0.3618 | 0.5648 | 599 |
| pool20 | full | mmr_0.7 | 0.0401 | 0.7990 | 0.4446 | 0.3684 | 0.5722 | 599 |
| pool20 | no_diversity | ga_s0 | 0.0267 | 0.7724 | 0.4734 | 0.4738 | 0.5774 | 599 |
| pool20 | no_diversity | greedy | 0.0384 | 0.7616 | 0.4415 | 0.3618 | 0.5648 | 599 |
| pool20 | no_diversity | mmr_0.7 | 0.0401 | 0.7990 | 0.4446 | 0.3684 | 0.5722 | 599 |
| pool20 | no_fresh | ga_s0 | 0.0317 | 0.8171 | 0.4799 | 0.3783 | 0.5803 | 599 |
| pool20 | no_fresh | greedy | 0.0384 | 0.7616 | 0.4415 | 0.3618 | 0.5648 | 599 |
| pool20 | no_fresh | mmr_0.7 | 0.0401 | 0.7990 | 0.4446 | 0.3684 | 0.5722 | 599 |
| pool20 | no_novelty | ga_s0 | 0.0317 | 0.8168 | 0.4465 | 0.4846 | 0.5823 | 599 |
| pool20 | no_novelty | greedy | 0.0384 | 0.7616 | 0.4415 | 0.3618 | 0.5648 | 599 |
| pool20 | no_novelty | mmr_0.7 | 0.0401 | 0.7990 | 0.4446 | 0.3684 | 0.5722 | 599 |
| pool50 | full | ga_s0 | 0.0234 | 0.8398 | 0.4923 | 0.5369 | 0.6440 | 599 |
| pool50 | full | greedy | 0.0384 | 0.7616 | 0.4415 | 0.3618 | 0.6156 | 599 |
| pool50 | full | mmr_0.7 | 0.0401 | 0.8118 | 0.4458 | 0.3740 | 0.6261 | 599 |

Exhaustive oracle on 12-candidate pools (K=5): 599 pools, mean fitness gap 0.0, max gap 0.0, optimal share 1.0.

MMR is a separately named baseline, not the GA. The oracle proves optimality only within each pool.
