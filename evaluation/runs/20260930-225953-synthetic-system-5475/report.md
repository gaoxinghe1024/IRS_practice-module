# System evaluation - synthetic data

In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). `*_cold` clears the recommendation cache before each call.

| endpoint | p50 | p95 | n |
|---|---|---|---|
| search | 2.9000 | 3.6000 | 30 |
| detail | 4.9000 | 6.2000 | 30 |
| similar_cold | 5.3000 | 6.7000 | 30 |
| similar_warm | 3.9000 | 5.5000 | 30 |
| home_cold | 131.8000 | 147.6000 | 30 |
| home_warm | 5.0000 | 5.5000 | 30 |
| collection_recs | 6.2000 | 7.5000 | 30 |
| chat_local | 7.0000 | 24.9000 | 30 |
| plan_preview | 152.3000 | 213.5000 | 30 |

Concurrency 5 (mixed reads): {'p50': 16.0, 'p95': 26.3, 'n': 60, 'errors': 0}

## Failure injection and robustness

| case | ok | status |
|---|---|---|
| llm_timeout | False | 200 |
| llm_rate_limited | False | 200 |
| llm_invalid_json | False | 200 |
| llm_empty | False | 200 |
| llm_refusal | False | 200 |
| dense_encoder_failure | True | 200 |
| long_input_1000_chars | True | 200 |
| over_limit_input_rejected | True | 422 |
| html_injection_is_data | True | 200 |
| extreme_filters | True | 200 |
| contradiction_clarified | True | – |
| concurrent_rating_writes | True | – |

LLM token cost is unknown / not applicable in local mode; nothing is reported as 0.
