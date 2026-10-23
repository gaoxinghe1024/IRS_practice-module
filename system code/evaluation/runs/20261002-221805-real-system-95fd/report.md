# System evaluation - real data

In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). `*_cold` clears the recommendation cache before each call.

| endpoint | p50 | p95 | n |
|---|---|---|---|
| search | 4.3000 | 9.6000 | 30 |
| detail | 6.7000 | 11.8000 | 30 |
| similar_cold | 8.9000 | 22.4000 | 30 |
| similar_warm | 3.2000 | 5.1000 | 30 |
| home_cold | 43.9000 | 49.5000 | 30 |
| home_warm | 4.1000 | 4.9000 | 30 |
| collection_recs | 131.4000 | 135.9000 | 30 |
| chat_local | 9.1000 | 12.0000 | 30 |
| plan_preview | 239.7000 | 313.9000 | 30 |

Concurrency 5 (mixed reads): {'p50': 24.0, 'p95': 44.4, 'n': 60, 'errors': 0}

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
