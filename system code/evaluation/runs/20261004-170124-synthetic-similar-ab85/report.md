# Similar titles (algorithm 1) - synthetic data

Dataset `synthetic-7a7d5d6a74`. Candidate pool: 175 served, recommendable titles. K=5. Each label source is evaluated separately. `split=val/test` applies to the source the logistic model was trained on (query groups kept franchise-disjoint); other sources are evaluated on all queries (`all`).

> Synthetic generator labels validate the pipeline only; they partly reuse the tag structure the attribute scorers see, so they are not an unbiased quality estimate.

| label_source | split | scorer | hits@5 | recall@5 | ndcg@5 | coverage | queries |
|---|---|---|---|---|---|---|---|
| synthetic_generator | train | A_weighted_attributes | 0.6486 | 0.3086 | 0.2832 | 0.6703 | 37 |
| synthetic_generator | train | B_logistic_similarity | 0.7568 | 0.3649 | 0.3115 | 0.7443 | 37 |
| synthetic_generator | train | C_bge_attribute_embedding | 0.4054 | 0.1396 | 0.1584 | 0.7822 | 37 |
| synthetic_generator | train | D_tag_semantic_hybrid | 0.6486 | 0.3288 | 0.3016 | 0.6824 | 37 |
| synthetic_generator | train | variant_semantic_synopsis | 0.1351 | 0.0338 | 0.0640 | 0.7127 | 37 |
| synthetic_generator | val | A_weighted_attributes | 0.7000 | 0.2917 | 0.3230 | 0.6340 | 10 |
| synthetic_generator | val | B_logistic_similarity | 0.6000 | 0.2833 | 0.3104 | 0.7380 | 10 |
| synthetic_generator | val | C_bge_attribute_embedding | 0.6000 | 0.1667 | 0.1510 | 0.7570 | 10 |
| synthetic_generator | val | D_tag_semantic_hybrid | 0.7000 | 0.2917 | 0.3194 | 0.6530 | 10 |
| synthetic_generator | val | variant_semantic_synopsis | 0.3000 | 0.1083 | 0.1099 | 0.7440 | 10 |
| synthetic_generator | test | A_weighted_attributes | 0.7692 | 0.4103 | 0.4874 | 0.6431 | 13 |
| synthetic_generator | test | B_logistic_similarity | 0.6923 | 0.3462 | 0.3418 | 0.7161 | 13 |
| synthetic_generator | test | C_bge_attribute_embedding | 0.5385 | 0.2628 | 0.3561 | 0.7346 | 13 |
| synthetic_generator | test | D_tag_semantic_hybrid | 0.7692 | 0.3910 | 0.4722 | 0.6500 | 13 |
| synthetic_generator | test | variant_semantic_synopsis | 0.0769 | 0.0256 | 0.0287 | 0.6708 | 13 |
| franchise_sanity | all | A_weighted_attributes | 1.0000 | 1.0000 | 0.9631 | 0.7025 | 20 |
| franchise_sanity | all | B_logistic_similarity | 1.0000 | 1.0000 | 0.9446 | 0.7645 | 20 |
| franchise_sanity | all | C_bge_attribute_embedding | 1.0000 | 1.0000 | 0.9631 | 0.7940 | 20 |
| franchise_sanity | all | D_tag_semantic_hybrid | 1.0000 | 1.0000 | 0.9631 | 0.7140 | 20 |
| franchise_sanity | all | variant_semantic_synopsis | 0.9000 | 0.9000 | 0.8069 | 0.7445 | 20 |

## Replacement rule on validation (Hits@5 +2pp and NDCG@5 not lower than A)

| source | candidate | val_queries | delta_hits@5 | delta_ndcg@5 | replace_default |
|---|---|---|---|---|---|
| synthetic_generator | B_logistic_similarity | 10 | -0.1000 | -0.0126 | no decision (n=10 < 20) |
| synthetic_generator | C_bge_attribute_embedding | 10 | -0.1000 | -0.1720 | no decision (n=10 < 20) |
| synthetic_generator | D_tag_semantic_hybrid | 10 | 0.0000 | -0.0036 | no decision (n=10 < 20) |
| synthetic_generator | variant_semantic_synopsis | 10 | -0.4000 | -0.2131 | no decision (n=10 < 20) |

The served default changes only by configuration; this table records what the rule says, and it makes no decision below the minimum number of validation queries.

`coverage` is the mean metadata coverage (share of weighted attribute dimensions that were comparable) of the returned titles; catalog coverage per scorer is in metrics.json.

`franchise_sanity` asks whether a title's confirmed sequel/prequel appears in its top 5. It is a sanity check of franchise awareness, not a general similarity benchmark: the synopsis variant does well here because franchise entries share character names.

