# System evaluation - real data

In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). `*_cold` clears the recommendation cache before each call.

| endpoint | p50 | p95 | n |
|---|---|---|---|
| search | 3.3000 | 9.7000 | 30 |
| detail | 6.8000 | 8.8000 | 30 |
| similar_cold | 9.7000 | 12.1000 | 30 |
| similar_warm | 3.0000 | 3.5000 | 30 |
| home_cold | 43.1000 | 45.8000 | 30 |
| home_warm | 3.7000 | 4.5000 | 30 |
| collection_recs | 128.9000 | 134.5000 | 30 |
| chat_local | 8.9000 | 12.2000 | 30 |
| plan_preview | 232.1000 | 330.1000 | 30 |

Concurrency 5 (mixed reads): {'p50': 28.2, 'p95': 45.8, 'n': 60, 'errors': 0}

## Failure injection and robustness

| case | ok | status |
|---|---|---|
| llm_timeout | True | 200 |
| llm_rate_limited | True | 200 |
| llm_invalid_json | True | 200 |
| llm_empty | True | 200 |
| llm_refusal | True | 200 |
| dense_encoder_failure | True | 200 |
| long_input_1000_chars | True | 200 |
| over_limit_input_rejected | True | 422 |
| html_injection_is_data | True | 200 |
| extreme_filters | True | 200 |
| contradiction_clarified | True | – |
| concurrent_rating_writes | True | – |

LLM token cost is unknown / not applicable in local mode; nothing is reported as 0.
