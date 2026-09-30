# Chat-to-find (algorithm 4) - synthetic data

All answers here were produced in the configured answer mode (local evidence templates unless an LLM is configured). Each query ran in a fresh chat session on a throwaway copy of the activated database.

> Synthetic generator queries validate the pipeline only.

| set | queries | tsr | hits@5 | mrr@10 | compliant | fabricated_episode | intent_ok | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| labelled_synthetic_generator | 41 | 0.9512 | 0.9459 | 0.8401 | 1.0000 | 0.0000 | 1.0000 | 23.2645 |

## Retrieval ablation (descriptive labelled queries, no constraints)

| retrieval | queries | hits@5 | mrr@10 |
|---|---|---|---|
| lexical | 16 | 0.7500 | 0.6319 |
| dense | 16 | 1.0000 | 0.9375 |
| hybrid | 16 | 0.9375 | 0.7552 |

## Plain LLM vs RAG: blocked_by_credentials

No LLM configured: chat uses deterministic parsing and local evidence templates (local mode).

