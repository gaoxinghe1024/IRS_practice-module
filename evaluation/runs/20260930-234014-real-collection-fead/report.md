# Collection recommendations (algorithm 2) - real data

Dataset `real-89bfdea00b`. Leave-one-out cases; K=8; the rule layer (served, recommendable, not already in the visible collection) is identical for every scorer, positives and negatives.

| source | scorer | hits@8 | ndcg@8 | auc_full_pool | auc_sampled50 | cases |
|---|---|---|---|---|---|---|
| proxy_liked_history:val | manual_slides_v0 | 0.0118 | 0.0054 | 0.8300 | 0.8309 | 1443 |
| proxy_liked_history:val | collection_profile_v1 | 0.0076 | 0.0043 | 0.7911 | 0.7901 | 1443 |
| proxy_liked_history:val | gradient_boosting_v1_proxy | 0.0915 | 0.0533 | 0.9643 | 0.9638 | 1443 |
| proxy_liked_history:test | manual_slides_v0 | 0.0131 | 0.0063 | 0.8306 | 0.8315 | 1446 |
| proxy_liked_history:test | collection_profile_v1 | 0.0097 | 0.0046 | 0.7840 | 0.7842 | 1446 |
| proxy_liked_history:test | gradient_boosting_v1_proxy | 0.1093 | 0.0638 | 0.9628 | 0.9628 | 1446 |

## Replacement rule on validation (AUC >= 0.65 and Hits@8 +2pp vs collection_profile_v1)

| source | candidate | val_auc_sampled50 | delta_hits@8 | replace_default |
|---|---|---|---|---|
| proxy_liked_history:val | gradient_boosting_v1_proxy | 0.9638 | 0.0839 | True |

AUC against sampled unobserved titles treats them as negatives only for measurement; they are not confirmed dislikes. `manual_slides_v0` is the slides' formula kept verbatim as the baseline.

- note: No real collection events: collection scorers are compared on the proxy_liked_history benchmark (NOT a collection task).
