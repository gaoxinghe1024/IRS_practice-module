# Chat-to-find (algorithm 4) - synthetic data

All answers here were produced in the configured answer mode (local evidence templates unless an LLM is configured). Each query ran in a fresh chat session on a throwaway copy of the activated database.

> Synthetic generator queries validate the pipeline only.

| set | queries | tsr | hits@5 | mrr@10 | scene@3 | false_scene_claim | compliant | fabricated_episode | intent_ok | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| team_package_20260918_A_final_v1_4a2e38e69be5:scene | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | 1.0000 | 0.0000 | 1.0000 | 36.4823 |
| team_package_20260918_A_final_v1_4a2e38e69be5:semantic | 30 | 0.0000 | 0.0000 | 0.0000 | – | – | 0.9333 | 0.0000 | 1.0000 | 3.8730 |
| team_package_20260918_A_final_v1_4a2e38e69be5:constraint | 20 | 0.0000 | 0.0000 | 0.0000 | – | – | 1.0000 | 0.0000 | 1.0000 | 3.1451 |
| team_package_20260918_A_final_v1_4a2e38e69be5:unanswerable | 10 | 0.9000 | – | – | – | – | 0.9000 | 0.0000 | – | 10.1536 |

## Scene retrieval ablation (labelled scene queries, raw query text)

Item-level hits over all served titles; scene@k counts the expected scene record itself. The synopsis-only rows show what plot-level matching finds without the scene layer.

| corpus | retrieval | queries | hits@1 | hits@5 | mrr@10 | scene@1 | scene@3 |
|---|---|---|---|---|---|---|---|
| scene | lexical | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scene | dense | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scene | hybrid | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| synopsis | lexical | 40 | 0.0000 | 0.0000 | 0.0000 | – | – |
| synopsis | dense | 40 | 0.0000 | 0.0000 | 0.0000 | – | – |
| synopsis | hybrid | 40 | 0.0000 | 0.0000 | 0.0000 | – | – |
| scene+synopsis | lexical | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scene+synopsis | dense | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scene+synopsis | hybrid | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Plain LLM vs RAG: blocked_by_credentials

No LLM configured: chat uses deterministic parsing and local evidence templates (local mode).

