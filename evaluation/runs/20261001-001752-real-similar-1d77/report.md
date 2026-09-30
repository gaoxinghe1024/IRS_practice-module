# Similar titles (algorithm 1) - real data

Dataset `real-89bfdea00b`. Candidate pool: 12429 served, recommendable titles. K=5. Each label source is evaluated separately. `split=val/test` applies to the source the logistic model was trained on (query groups kept franchise-disjoint); other sources are evaluated on all queries (`all`).

| label_source | split | scorer | hits@5 | recall@5 | ndcg@5 | coverage | queries |
|---|---|---|---|---|---|---|---|
| mal_user_recommendations_weak | val | A_weighted_attributes | 0.0000 | 0.0000 | 0.0000 | 0.5100 | 1 |
| mal_user_recommendations_weak | val | C_bge_attribute_embedding | 1.0000 | 0.0500 | 0.1461 | 0.5500 | 1 |
| mal_user_recommendations_weak | val | D_tag_semantic_hybrid | 0.0000 | 0.0000 | 0.0000 | 0.5100 | 1 |
| mal_user_recommendations_weak | val | variant_semantic_synopsis | 0.0000 | 0.0000 | 0.0000 | 0.5500 | 1 |
| mal_user_recommendations_weak | train | A_weighted_attributes | 0.0000 | 0.0000 | 0.0000 | 0.5500 | 1 |
| mal_user_recommendations_weak | train | C_bge_attribute_embedding | 0.0000 | 0.0000 | 0.0000 | 0.5500 | 1 |
| mal_user_recommendations_weak | train | D_tag_semantic_hybrid | 0.0000 | 0.0000 | 0.0000 | 0.5500 | 1 |
| mal_user_recommendations_weak | train | variant_semantic_synopsis | 0.0000 | 0.0000 | 0.0000 | 0.5500 | 1 |
| franchise_sanity | all | A_weighted_attributes | 0.6522 | 0.5870 | 0.5212 | 0.5361 | 23 |
| franchise_sanity | all | C_bge_attribute_embedding | 0.4783 | 0.4348 | 0.2895 | 0.6661 | 23 |
| franchise_sanity | all | D_tag_semantic_hybrid | 0.6522 | 0.5870 | 0.4981 | 0.5404 | 23 |
| franchise_sanity | all | variant_semantic_synopsis | 0.8261 | 0.8261 | 0.7685 | 0.5578 | 23 |

## Replacement rule on validation (Hits@5 +2pp and NDCG@5 not lower than A)

| source | candidate | val_queries | delta_hits@5 | delta_ndcg@5 | replace_default |
|---|---|---|---|---|---|
| mal_user_recommendations_weak | C_bge_attribute_embedding | 1 | 1.0000 | 0.1461 | no decision (n=1 < 20) |
| mal_user_recommendations_weak | D_tag_semantic_hybrid | 1 | 0.0000 | 0.0000 | no decision (n=1 < 20) |
| mal_user_recommendations_weak | variant_semantic_synopsis | 1 | 0.0000 | 0.0000 | no decision (n=1 < 20) |

The served default changes only by configuration; this table records what the rule says, and it makes no decision below the minimum number of validation queries.

`coverage` is the mean metadata coverage (share of weighted attribute dimensions that were comparable) of the returned titles; catalog coverage per scorer is in metrics.json.

`franchise_sanity` asks whether a title's confirmed sequel/prequel appears in its top 5. It is a sanity check of franchise awareness, not a general similarity benchmark: the synopsis variant does well here because franchise entries share character names.

