# Viewing-plan optimisation - real data

120 offline users (train histories as profiles), 5 request types, complete-title mode, prerequisites allowed. Every plan was re-validated independently.

| request | method | fitness | feasible | utilization | titles | relevance | diversity | ms | n |
|---|---|---|---|---|---|---|---|---|---|
| b1500_t5_normal | ga | 0.8073 | 1.0000 | 0.9246 | 4.2333 | 0.7884 | 0.9030 | 188.7304 | 120 |
| b1500_t5_normal | greedy | 0.6486 | 1.0000 | 0.1487 | 5.0000 | 0.7892 | 0.6634 | 0.2637 | 120 |
| b300_t2_normal | ga | 0.8030 | 1.0000 | 0.8538 | 1.8417 | 0.7899 | 0.9153 | 122.6655 | 120 |
| b300_t2_normal | greedy | 0.6184 | 1.0000 | 0.1537 | 1.9833 | 0.7895 | 0.5624 | 0.2358 | 120 |
| b600_t3_normal | ga | 0.8121 | 1.0000 | 0.9040 | 2.5167 | 0.7900 | 0.9210 | 141.5024 | 120 |
| b600_t3_normal | greedy | 0.6363 | 1.0000 | 0.1544 | 3.0000 | 0.7894 | 0.6205 | 0.2459 | 120 |
| b900_t3_high | ga | 0.8304 | 1.0000 | 0.8582 | 2.4167 | 0.7894 | 0.9586 | 156.2629 | 120 |
| b900_t3_high | greedy | 0.6298 | 1.0000 | 0.1029 | 3.0000 | 0.7894 | 0.6205 | 0.2421 | 120 |
| b900_t5_low | ga | 0.8051 | 1.0000 | 0.9614 | 3.9500 | 0.7889 | 0.8755 | 173.1630 | 120 |
| b900_t5_low | greedy | 0.6563 | 1.0000 | 0.2369 | 4.9833 | 0.7892 | 0.6615 | 0.2625 | 120 |

Infeasible plans returned: 0.

Exact oracle on 12-unit pools: {'cases': 600, 'mean_gap': 0.00014259614370329076, 'max_gap': 0.05988719540548537, 'share_optimal': 0.99}.

GA mean fitness by seed: {'b300_t2_normal': [0.80298, 0.80254, 0.80297], 'b600_t3_normal': [0.81208, 0.81207, 0.81162], 'b900_t3_high': [0.83039, 0.8298, 0.82983], 'b1500_t5_normal': [0.80732, 0.80769, 0.80724], 'b900_t5_low': [0.80512, 0.80511, 0.80552]}.

Utilisation has a small weight in the objective on purpose: the plan should not be padded with low-relevance titles only to fill the budget.
