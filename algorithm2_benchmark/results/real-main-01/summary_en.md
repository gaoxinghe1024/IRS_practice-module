# Algorithm 2 — real-main-01 (level: main)

| Model | Users | HR@8 | NDCG@8 | MRR@8 | AUC | ΔHR@8 vs A0, pp [95% CI] | ΔHR@8 vs A1, pp [95% CI] | P95 service latency (ms) | Training time (s) | Model size (KB) | Rule violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 3000 | 0.0210 | 0.0107 | 0.0075 | 0.8699 | — | -1.10 [-1.60, -0.57] | 11.3 | 0 | 0 | 0 |
| A1 | 3000 | 0.0320 | 0.0179 | 0.0136 | 0.9082 | +1.10 [+0.57, +1.60] | — | 11.8 | 0 | 0 | 0 |
| B0 (B0_seed42) | 3000 | 0.0543 | 0.0316 | 0.0245 | 0.9213 | +3.33 [+2.50, +4.17] | +2.23 [+1.53, +2.97] | 110.1 | 5.1 | 1699 | 0 |
| B1 (B1_seed42) | 3000 | 0.1317 | 0.0723 | 0.0538 | 0.9527 | +11.07 [+9.83, +12.30] | +9.97 [+8.73, +11.17] | 153.2 | 5.4 | 1714 | 0 |
| C0 (C0_seed42) | 3000 | 0.1083 | 0.0621 | 0.0478 | 0.9640 | +8.73 [+7.53, +9.93] | +7.63 [+6.53, +8.77] | 1.0 | 26.5 | 482 | 0 |

Primary set: rule R0, main group (visible history >= 5), more than 8 candidates, target eligible; full-catalogue ranking; user-macro averages. Learned models: deploy seed 42; seed means are in final_report.md.
Label: proxy (personal rating >= 8 in the Kaggle MyAnimeList 2023 snapshot), not real collections.

Pre-selected on validation: **B1** · confirmed on test: **True** · default scorer: **B1**
Reason: B1 passed every pre-registered confirmation check on test
