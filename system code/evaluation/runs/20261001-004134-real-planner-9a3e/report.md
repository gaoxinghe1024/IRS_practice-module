# Viewing-plan optimisation - real data

120 offline users (train histories as profiles), 5 request types, complete-title mode, prerequisites allowed. Every plan was re-validated independently.

| request | method | fitness | feasible | utilization | titles | relevance | diversity | ms | n |
|---|---|---|---|---|---|---|---|---|---|
| b1500_t5_normal | ga | 0.8322 | 1.0000 | 0.9520 | 3.6000 | 0.7904 | 0.9329 | 185.9118 | 120 |
| b1500_t5_normal | greedy | 0.6309 | 1.0000 | 0.1051 | 5.0000 | 0.7896 | 0.6620 | 0.2130 | 120 |
| b300_t2_normal | ga | 0.8235 | 1.0000 | 0.9269 | 1.6833 | 0.7899 | 0.9024 | 110.6390 | 120 |
| b300_t2_normal | greedy | 0.6094 | 1.0000 | 0.1212 | 2.0000 | 0.7897 | 0.6396 | 0.1816 | 120 |
| b600_t3_normal | ga | 0.8296 | 1.0000 | 0.9326 | 2.5750 | 0.7901 | 0.9087 | 140.5311 | 120 |
| b600_t3_normal | greedy | 0.6142 | 1.0000 | 0.1134 | 3.0000 | 0.7897 | 0.6246 | 0.1909 | 120 |
| b900_t3_high | ga | 0.8446 | 1.0000 | 0.8981 | 2.4917 | 0.7899 | 0.9494 | 154.3046 | 120 |
| b900_t3_high | greedy | 0.6123 | 1.0000 | 0.0756 | 3.0000 | 0.7897 | 0.6246 | 0.1914 | 120 |
| b900_t5_low | ga | 0.8224 | 1.0000 | 0.9757 | 3.8417 | 0.7901 | 0.8810 | 173.8537 | 120 |
| b900_t5_low | greedy | 0.6352 | 1.0000 | 0.1751 | 5.0000 | 0.7896 | 0.6620 | 0.2134 | 120 |

Infeasible plans returned: 0.

Exact oracle on 12-unit pools: {'cases': 600, 'mean_gap': 0.000696223912962214, 'max_gap': 0.06353417424972063, 'share_optimal': 0.9533333333333334}.

GA mean fitness by seed: {'b300_t2_normal': [0.8235, 0.82307, 0.82394], 'b600_t3_normal': [0.82964, 0.82855, 0.82916], 'b900_t3_high': [0.84456, 0.84409, 0.84473], 'b1500_t5_normal': [0.83222, 0.83094, 0.82942], 'b900_t5_low': [0.82241, 0.82221, 0.8222]}.

Utilisation has a small weight in the objective on purpose: the plan should not be padded with low-relevance titles only to fill the budget.
