# Similar titles (algorithm 1) - real data

Dataset `real-034ebd6a74`. Candidate pool: 13798 served, recommendable titles. K=5. Each label source is evaluated separately. `split=val/test` applies to the source the logistic model was trained on (query groups kept franchise-disjoint); other sources are evaluated on all queries (`all`).

| label_source | split | scorer | hits@5 | recall@5 | ndcg@5 | coverage | queries |
|---|---|---|---|---|---|---|---|
| mal_user_recommendations_weak | train | A_weighted_attributes | 0.0943 | 0.0214 | 0.0155 | 0.5881 | 53 |
| mal_user_recommendations_weak | train | B_logistic_similarity | 0.2641 | 0.0701 | 0.0596 | 0.8313 | 53 |
| mal_user_recommendations_weak | train | C_bge_attribute_embedding | 0.1887 | 0.0453 | 0.0407 | 0.8260 | 53 |
| mal_user_recommendations_weak | train | D_tag_semantic_hybrid | 0.0755 | 0.0151 | 0.0136 | 0.6309 | 53 |
| mal_user_recommendations_weak | train | variant_semantic_synopsis | 0.1132 | 0.0289 | 0.0270 | 0.7742 | 53 |
| mal_user_recommendations_weak | val | A_weighted_attributes | 0.0490 | 0.0103 | 0.0090 | 0.5717 | 102 |
| mal_user_recommendations_weak | val | B_logistic_similarity | 0.1667 | 0.0382 | 0.0392 | 0.8564 | 102 |
| mal_user_recommendations_weak | val | C_bge_attribute_embedding | 0.1765 | 0.0410 | 0.0411 | 0.8533 | 102 |
| mal_user_recommendations_weak | val | D_tag_semantic_hybrid | 0.0686 | 0.0162 | 0.0136 | 0.6166 | 102 |
| mal_user_recommendations_weak | val | variant_semantic_synopsis | 0.1373 | 0.0293 | 0.0267 | 0.8072 | 102 |
| mal_user_recommendations_weak | test | A_weighted_attributes | 0.0812 | 0.0192 | 0.0192 | 0.5858 | 234 |
| mal_user_recommendations_weak | test | B_logistic_similarity | 0.1966 | 0.0490 | 0.0518 | 0.8756 | 234 |
| mal_user_recommendations_weak | test | C_bge_attribute_embedding | 0.1624 | 0.0357 | 0.0376 | 0.8741 | 234 |
| mal_user_recommendations_weak | test | D_tag_semantic_hybrid | 0.0769 | 0.0182 | 0.0190 | 0.6162 | 234 |
| mal_user_recommendations_weak | test | variant_semantic_synopsis | 0.1026 | 0.0214 | 0.0227 | 0.8226 | 234 |
| franchise_sanity | all | A_weighted_attributes | 0.4249 | 0.3712 | 0.3067 | 0.5605 | 4507 |
| franchise_sanity | all | B_logistic_similarity | 0.6443 | 0.5892 | 0.4984 | 0.7899 | 4507 |
| franchise_sanity | all | C_bge_attribute_embedding | 0.6299 | 0.5716 | 0.4798 | 0.7871 | 4507 |
| franchise_sanity | all | D_tag_semantic_hybrid | 0.4886 | 0.4333 | 0.3601 | 0.5901 | 4507 |
| franchise_sanity | all | variant_semantic_synopsis | 0.6963 | 0.6431 | 0.5600 | 0.7327 | 4507 |

## Replacement rule on validation (Hits@5 +2pp and NDCG@5 not lower than A)

| source | candidate | val_queries | delta_hits@5 | delta_ndcg@5 | replace_default |
|---|---|---|---|---|---|
| mal_user_recommendations_weak | B_logistic_similarity | 102 | 0.1177 | 0.0302 | True |
| mal_user_recommendations_weak | C_bge_attribute_embedding | 102 | 0.1274 | 0.0322 | True |
| mal_user_recommendations_weak | D_tag_semantic_hybrid | 102 | 0.0196 | 0.0046 | False |
| mal_user_recommendations_weak | variant_semantic_synopsis | 102 | 0.0882 | 0.0178 | True |

The served default changes only by configuration; this table records what the rule says, and it makes no decision below the minimum number of validation queries.

`coverage` is the mean metadata coverage (share of weighted attribute dimensions that were comparable) of the returned titles; catalog coverage per scorer is in metrics.json.

`franchise_sanity` asks whether a title's confirmed sequel/prequel appears in its top 5. It is a sanity check of franchise awareness, not a general similarity benchmark: the synopsis variant does well here because franchise entries share character names.

