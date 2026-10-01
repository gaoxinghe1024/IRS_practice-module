# Collection recommendations (algorithm 2) - synthetic data

Dataset `synthetic-7a7d5d6a74`. Leave-one-out cases; K=8; the rule layer (served, recommendable, not already in the visible collection) is identical for every scorer, positives and negatives.

> Synthetic collections come from a generator process; results validate the implementation only.

| source | scorer | hits@8 | ndcg@8 | auc_full_pool | auc_sampled50 | cases |
|---|---|---|---|---|---|---|
| synthetic_generator:val | manual_slides_v0 | 0.0566 | 0.0250 | 0.5852 | 0.5782 | 265 |
| synthetic_generator:val | collection_profile_v1 | 0.0679 | 0.0341 | 0.5997 | 0.5997 | 265 |
| synthetic_generator:val | gradient_boosting_v1 | 0.1245 | 0.0561 | 0.6271 | 0.6244 | 265 |
| synthetic_generator:val | two_tower_v1 | 0.0943 | 0.0424 | 0.5817 | 0.5841 | 265 |
| synthetic_generator:test | manual_slides_v0 | 0.0691 | 0.0445 | 0.6062 | 0.6089 | 217 |
| synthetic_generator:test | collection_profile_v1 | 0.0876 | 0.0405 | 0.5859 | 0.5796 | 217 |
| synthetic_generator:test | gradient_boosting_v1 | 0.0968 | 0.0481 | 0.6133 | 0.6091 | 217 |
| synthetic_generator:test | two_tower_v1 | 0.0599 | 0.0286 | 0.5394 | 0.5401 | 217 |

## Replacement rule on validation (AUC >= 0.65 and Hits@8 +2pp vs collection_profile_v1)

| source | candidate | val_auc_sampled50 | delta_hits@8 | replace_default |
|---|---|---|---|---|
| synthetic_generator:val | gradient_boosting_v1 | 0.6244 | 0.0566 | False |
| synthetic_generator:val | two_tower_v1 | 0.5841 | 0.0264 | False |

AUC against sampled unobserved titles treats them as negatives only for measurement; they are not confirmed dislikes. `manual_slides_v0` is the slides' formula kept verbatim as the baseline.

