# Viewing-plan optimisation - real data

120 offline users (train histories as profiles), 5 request types, complete-title mode, prerequisites allowed. Every plan was re-validated independently.

| request | method | fitness | feasible | utilization | titles | relevance | diversity | ms | n |
|---|---|---|---|---|---|---|---|---|---|
| b1500_t5_normal | ga | 0.8014 | 1.0000 | 0.9174 | 4.2500 | 0.7884 | 0.9015 | 187.2582 | 120 |
| b1500_t5_normal | greedy | 0.6363 | 1.0000 | 0.1396 | 5.0000 | 0.7892 | 0.6646 | 0.2517 | 120 |
| b300_t2_normal | ga | 0.7961 | 1.0000 | 0.8485 | 1.8500 | 0.7898 | 0.9110 | 116.7432 | 120 |
| b300_t2_normal | greedy | 0.6051 | 1.0000 | 0.1442 | 1.9833 | 0.7895 | 0.5681 | 0.2242 | 120 |
| b600_t3_normal | ga | 0.8092 | 1.0000 | 0.9165 | 2.4667 | 0.7899 | 0.9186 | 135.6031 | 120 |
| b600_t3_normal | greedy | 0.6231 | 1.0000 | 0.1466 | 3.0000 | 0.7894 | 0.6234 | 0.2314 | 120 |
| b900_t3_high | ga | 0.8265 | 1.0000 | 0.8436 | 2.4083 | 0.7894 | 0.9584 | 152.5556 | 120 |
| b900_t3_high | greedy | 0.6189 | 1.0000 | 0.0977 | 3.0000 | 0.7894 | 0.6234 | 0.2316 | 120 |
| b900_t5_low | ga | 0.8003 | 1.0000 | 0.9554 | 3.9083 | 0.7892 | 0.8788 | 163.1512 | 120 |
| b900_t5_low | greedy | 0.6429 | 1.0000 | 0.2281 | 4.9917 | 0.7892 | 0.6635 | 0.2526 | 120 |

Infeasible plans returned: 0.

Exact oracle on 12-unit pools: {'cases': 600, 'mean_gap': 0.00015679460356928207, 'max_gap': 0.05413416136160032, 'share_optimal': 0.9866666666666667}.

GA mean fitness by seed: {'b300_t2_normal': [0.79608, 0.79539, 0.79613], 'b600_t3_normal': [0.8092, 0.80864, 0.80975], 'b900_t3_high': [0.82653, 0.82632, 0.82625], 'b1500_t5_normal': [0.80136, 0.80207, 0.80145], 'b900_t5_low': [0.80034, 0.80026, 0.80028]}.

Utilisation has a small weight in the objective on purpose: the plan should not be padded with low-relevance titles only to fill the budget.
