# Collection recommendations (algorithm 2) - real data

Dataset `real-034ebd6a74`. Leave-one-out cases; K=8; the rule layer (served, recommendable, not already in the visible collection) is identical for every scorer, positives and negatives.

| source | scorer | hits@8 | ndcg@8 | auc_full_pool | auc_sampled50 | cases |
|---|---|---|---|---|---|---|
| proxy_liked_history:val | manual_slides_v0 | 0.0166 | 0.0086 | 0.8693 | 0.8674 | 1445 |
| proxy_liked_history:val | collection_profile_v1 | 0.0055 | 0.0037 | 0.8266 | 0.8276 | 1445 |
| proxy_liked_history:val | gradient_boosting_v1_proxy | 0.1114 | 0.0660 | 0.9692 | 0.9694 | 1445 |
| proxy_liked_history:test | manual_slides_v0 | 0.0235 | 0.0139 | 0.8606 | 0.8616 | 1446 |
| proxy_liked_history:test | collection_profile_v1 | 0.0090 | 0.0040 | 0.8218 | 0.8220 | 1446 |
| proxy_liked_history:test | gradient_boosting_v1_proxy | 0.1155 | 0.0724 | 0.9670 | 0.9670 | 1446 |

## Replacement rule on validation (AUC >= 0.65 and Hits@8 +2pp vs collection_profile_v1)

| source | candidate | val_auc_sampled50 | delta_hits@8 | replace_default |
|---|---|---|---|---|
| proxy_liked_history:val | gradient_boosting_v1_proxy | 0.9694 | 0.1059 | n/a (proxy benchmark) |

AUC against sampled unobserved titles treats them as negatives only for measurement; they are not confirmed dislikes. `manual_slides_v0` is the slides' formula kept verbatim as the baseline.

GBDT permutation importance on validation (gbdt_proxy_training): train_popularity 0.305, studio_affinity 0.010, catalog_score 0.004, genre_affinity 0.003.

On the proxy benchmark the GBDT's advantage comes almost entirely from train popularity: the hidden titles are titles the user rated >= 8, which are mostly popular. This says nothing about matching a collection's taste, so the proxy result does not change the collection default.

- note: No real collection events: collection scorers are compared on the proxy_liked_history benchmark (NOT a collection task).
