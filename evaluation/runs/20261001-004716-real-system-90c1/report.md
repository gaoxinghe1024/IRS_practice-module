# System evaluation - real data

In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). `*_cold` clears the recommendation cache before each call.

| endpoint | p50 | p95 | n |
|---|---|---|---|
| search | 3.0000 | 4.7000 | 30 |
| detail | 4.1000 | 5.4000 | 30 |
| similar_cold | 6.2000 | 7.2000 | 30 |
| similar_warm | 2.6000 | 3.1000 | 30 |
| home_cold | 36.9000 | 40.3000 | 30 |
| home_warm | 3.3000 | 4.1000 | 30 |
| collection_recs | 6.8000 | 9.0000 | 30 |
| chat_local | 6.0000 | 38.8000 | 30 |
| plan_preview | 187.7000 | 196.5000 | 30 |

Concurrency 5 (mixed reads): {'p50': 18.7, 'p95': 34.5, 'n': 60, 'errors': 0}

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
