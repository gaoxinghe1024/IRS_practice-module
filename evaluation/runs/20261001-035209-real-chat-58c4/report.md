# Chat-to-find (algorithm 4) - real data

All answers here were produced in the configured answer mode (local evidence templates unless an LLM is configured). Each query ran in a fresh chat session on a throwaway copy of the activated database.

> No human-annotated RAG query set was provided (data/evaluation/rag_queries.jsonl). The `auto_*` sets are generated from catalogue facts and test robustness, constraint compliance and refusal behaviour; they are not relevance judgements.

| set | queries | tsr | hits@5 | mrr@10 | compliant | fabricated_episode | intent_ok | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| auto_titles | 312 | 0.9167 | 0.9167 | 0.9120 | 1.0000 | 0.0000 | 0.9519 | 115.8376 |
| auto_constraints | 38 | 1.0000 | – | – | 1.0000 | 0.0000 | 1.0000 | 35.0135 |
| auto_relations | 17 | 1.0000 | 1.0000 | 0.4461 | 1.0000 | 0.0000 | 1.0000 | 180.3349 |
| auto_no_episode | 30 | 1.0000 | – | – | 1.0000 | 0.0000 | 1.0000 | 2.2854 |

## Plain LLM vs RAG: blocked_by_credentials

No LLM configured: chat uses deterministic parsing and local evidence templates (local mode).

