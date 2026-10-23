# Viewing-plan optimisation - synthetic data

60 offline users (train histories as profiles), 5 request types, complete-title mode, prerequisites allowed. Every plan was re-validated independently.

> Synthetic data: validates optimiser correctness and feasibility, not plan quality.

| request | method | fitness | feasible | utilization | titles | relevance | diversity | ms | n |
|---|---|---|---|---|---|---|---|---|---|
| b1500_t5_normal | ga | 0.8139 | 1.0000 | 0.9672 | 4.0667 | 0.8023 | 0.9207 | 175.0089 | 60 |
| b1500_t5_normal | greedy | 0.6637 | 1.0000 | 0.1800 | 5.0000 | 0.8011 | 0.7803 | 0.2106 | 60 |
| b300_t2_normal | ga | 0.7964 | 1.0000 | 0.8761 | 1.7833 | 0.8015 | 0.9124 | 107.4853 | 60 |
| b300_t2_normal | greedy | 0.6554 | 1.0000 | 0.2106 | 2.0000 | 0.8004 | 0.8260 | 0.1826 | 60 |
| b600_t3_normal | ga | 0.8171 | 1.0000 | 0.9414 | 2.3167 | 0.8003 | 0.9357 | 152.2539 | 60 |
| b600_t3_normal | greedy | 0.6631 | 1.0000 | 0.2052 | 3.0000 | 0.8005 | 0.8013 | 0.1876 | 60 |
| b900_t3_high | ga | 0.8407 | 1.0000 | 0.9202 | 2.3500 | 0.8030 | 0.9617 | 140.1693 | 60 |
| b900_t3_high | greedy | 0.6752 | 1.0000 | 0.1368 | 3.0000 | 0.8005 | 0.8013 | 0.1882 | 60 |
| b900_t5_low | ga | 0.8094 | 1.0000 | 0.9764 | 3.2833 | 0.8029 | 0.9194 | 152.1767 | 60 |
| b900_t5_low | greedy | 0.6641 | 1.0000 | 0.2999 | 5.0000 | 0.8011 | 0.7803 | 0.2105 | 60 |

Infeasible plans returned: 0.

Exact oracle on 12-unit pools: {'cases': 300, 'mean_gap': 0.0010479001108400089, 'max_gap': 0.07391744181786375, 'share_optimal': 0.9133333333333333}.

GA mean fitness by seed: {'b300_t2_normal': [0.7964, 0.79605, 0.79604], 'b600_t3_normal': [0.81706, 0.81475, 0.81734], 'b900_t3_high': [0.84075, 0.84055, 0.84037], 'b1500_t5_normal': [0.8139, 0.81304, 0.8134], 'b900_t5_low': [0.80943, 0.8097, 0.80917]}.

Utilisation has a small weight in the objective on purpose: the plan should not be padded with low-relevance titles only to fill the budget.
