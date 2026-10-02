# Algorithm 2 — real-smoke-01 (level: smoke)

| Model | Users | HR@8 | NDCG@8 | MRR@8 | AUC | ΔHR@8 vs A0, pp [95% CI] | ΔHR@8 vs A1, pp [95% CI] | P95 service latency (ms) | Training time (s) | Model size (KB) | Rule violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 150 | 0.0200 | 0.0142 | 0.0122 | 0.8819 | — | -1.33 [-4.00, +1.33] | 10.5 | 0 | 0 | 0 |
| A1 | 150 | 0.0333 | 0.0181 | 0.0130 | 0.9163 | +1.33 [-1.33, +4.00] | — | 10.2 | 0 | 0 | 0 |
| B0 (B0_seed42) | 150 | 0.0467 | 0.0265 | 0.0205 | 0.9237 | +2.67 [-1.33, +7.33] | +1.33 [-2.00, +5.33] | 52.1 | 0.8 | 483 | 0 |
| B1 (B1_seed42) | 150 | 0.1000 | 0.0558 | 0.0423 | 0.9570 | +8.00 [+2.67, +13.33] | +6.67 [+2.00, +11.33] | 74.5 | 1.0 | 436 | 0 |
| C0 (C0_seed42) | 150 | 0.0667 | 0.0290 | 0.0172 | 0.9317 | +4.67 [+0.67, +8.67] | +3.33 [-1.33, +8.67] | 0.9 | 1.6 | 482 | 0 |

Primary set: rule R0, main group (visible history >= 5), more than 8 candidates, target eligible; full-catalogue ranking; user-macro averages. Learned models: deploy seed 42; seed means are in final_report.md.
Label: proxy (personal rating >= 8 in the Kaggle MyAnimeList 2023 snapshot), not real collections.

Pre-selected on validation: **C0** · confirmed on test: **False** · default scorer: **A1**
Reason: C0 did not pass ci_lower_vs_A1_above_zero; the pre-registered fallback A1 is deployed
