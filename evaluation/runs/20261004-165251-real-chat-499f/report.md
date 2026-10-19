# Chat-to-find (algorithm 4) - real data

All answers here were produced in the configured answer mode (local evidence templates unless an LLM is configured). Each query ran in a fresh chat session on a throwaway copy of the activated database.

> The `team_package_20260918_A_final_v1_4a2e38e69be5:*` sets are the team data package's evaluation queries. Scene queries are anchored to one scene record each (checked by an AI agent against cited sources, not by a person); semantic and constraint queries count every catalogue title that satisfies the stated filters as relevant, so Hits@5 there mostly measures constraint compliance. They are not independent human relevance judgements.

> The `auto_*` sets are generated from catalogue facts and test robustness, constraint compliance and refusal behaviour; they are not relevance judgements.

| set | queries | tsr | hits@5 | mrr@10 | scene@3 | false_scene_claim | compliant | fabricated_episode | intent_ok | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| team_package_20260918_A_final_v1_4a2e38e69be5:scene | 40 | 1.0000 | 1.0000 | 0.9333 | 1.0000 | – | 1.0000 | 0.0000 | 1.0000 | 109.7049 |
| team_package_20260918_A_final_v1_4a2e38e69be5:semantic | 30 | 1.0000 | 1.0000 | 0.9444 | – | – | 1.0000 | 0.0000 | 1.0000 | 8.0142 |
| team_package_20260918_A_final_v1_4a2e38e69be5:constraint | 20 | 1.0000 | 1.0000 | 0.9750 | – | – | 1.0000 | 0.0000 | 1.0000 | 7.4589 |
| team_package_20260918_A_final_v1_4a2e38e69be5:unanswerable | 10 | 1.0000 | – | – | – | – | 1.0000 | 0.0000 | – | 5.7874 |
| auto_titles | 320 | 0.9688 | 0.9688 | 0.9688 | – | – | 1.0000 | 0.0000 | 0.9656 | 144.5338 |
| auto_constraints | 38 | 1.0000 | – | – | – | – | 1.0000 | 0.0000 | 1.0000 | 7.0820 |
| auto_relations | 60 | 1.0000 | 1.0000 | 0.5000 | – | – | 1.0000 | 0.0000 | 1.0000 | 271.3061 |
| auto_no_episode | 30 | 1.0000 | – | – | – | – | 1.0000 | 0.0000 | 1.0000 | 89.7195 |
| auto_scene_outside | 60 | 0.9167 | 0.9167 | 0.8686 | – | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 226.6025 |
| auto_scene_mismatch | 40 | 1.0000 | – | – | – | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 63.6191 |

## Scene retrieval ablation (labelled scene queries, raw query text)

Item-level hits over all served titles; scene@k counts the expected scene record itself. The synopsis-only rows show what plot-level matching finds without the scene layer.

| corpus | retrieval | queries | hits@1 | hits@5 | mrr@10 | scene@1 | scene@3 |
|---|---|---|---|---|---|---|---|
| scene | lexical | 40 | 1.0000 | 1.0000 | 1.0000 | 0.9750 | 1.0000 |
| scene | dense | 40 | 0.9500 | 1.0000 | 0.9688 | 0.9500 | 0.9750 |
| scene | hybrid | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| synopsis | lexical | 40 | 0.2750 | 0.4000 | 0.3217 | – | – |
| synopsis | dense | 40 | 0.1250 | 0.2000 | 0.1744 | – | – |
| synopsis | hybrid | 40 | 0.3500 | 0.4250 | 0.3809 | – | – |
| scene+synopsis | lexical | 40 | 1.0000 | 1.0000 | 1.0000 | 0.9750 | 1.0000 |
| scene+synopsis | dense | 40 | 0.5250 | 0.7750 | 0.6190 | 0.9000 | 0.9000 |
| scene+synopsis | hybrid | 40 | 0.8250 | 0.9500 | 0.8836 | 0.9750 | 1.0000 |

## Plain LLM vs RAG: blocked_by_credentials

No LLM configured: chat uses deterministic parsing and local evidence templates (local mode).

