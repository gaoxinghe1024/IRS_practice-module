# Viewing-plan optimisation - synthetic data

60 offline users (train histories as profiles), 5 request types, complete-title mode, prerequisites allowed. Every plan was re-validated independently.

> Synthetic data: validates optimiser correctness and feasibility, not plan quality.

| request | method | fitness | feasible | utilization | titles | relevance | diversity | ms | n |
|---|---|---|---|---|---|---|---|---|---|
| b1500_t5_normal | ga | 0.8498 | 1.0000 | 0.9681 | 4.0167 | 0.8893 | 0.9233 | 198.4190 | 60 |
| b1500_t5_normal | greedy | 0.6939 | 1.0000 | 0.1771 | 5.0000 | 0.8703 | 0.7790 | 0.2344 | 60 |
| b300_t2_normal | ga | 0.8321 | 1.0000 | 0.8857 | 1.7833 | 0.8827 | 0.9131 | 124.9413 | 60 |
| b300_t2_normal | greedy | 0.6837 | 1.0000 | 0.2043 | 2.0000 | 0.8692 | 0.8166 | 0.2023 | 60 |
| b600_t3_normal | ga | 0.8513 | 1.0000 | 0.9147 | 2.1833 | 0.8839 | 0.9351 | 143.0927 | 60 |
| b600_t3_normal | greedy | 0.6882 | 1.0000 | 0.2000 | 3.0000 | 0.8666 | 0.8016 | 0.2239 | 60 |
| b900_t3_high | ga | 0.8710 | 1.0000 | 0.8979 | 2.2333 | 0.8866 | 0.9639 | 166.0363 | 60 |
| b900_t3_high | greedy | 0.6972 | 1.0000 | 0.1333 | 3.0000 | 0.8666 | 0.8016 | 0.2163 | 60 |
| b900_t5_low | ga | 0.8477 | 1.0000 | 0.9727 | 3.0667 | 0.8848 | 0.9209 | 169.9784 | 60 |
| b900_t5_low | greedy | 0.6976 | 1.0000 | 0.2952 | 5.0000 | 0.8703 | 0.7790 | 0.2426 | 60 |

Infeasible plans returned: 0.

Exact oracle on 12-unit pools: {'cases': 300, 'mean_gap': 0.0007532325046540811, 'max_gap': 0.05685289865321563, 'share_optimal': 0.9466666666666667}.

GA mean fitness by seed: {'b300_t2_normal': [0.83215, 0.83093, 0.83003], 'b600_t3_normal': [0.85131, 0.8536, 0.85581], 'b900_t3_high': [0.87096, 0.8705, 0.87083], 'b1500_t5_normal': [0.84978, 0.849, 0.84831], 'b900_t5_low': [0.84771, 0.84702, 0.84713]}.

Utilisation has a small weight in the objective on purpose: the plan should not be padded with low-relevance titles only to fill the budget.
