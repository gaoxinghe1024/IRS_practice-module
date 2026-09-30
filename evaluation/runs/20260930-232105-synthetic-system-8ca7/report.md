# System evaluation - synthetic data

In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). `*_cold` clears the recommendation cache before each call.

| endpoint | p50 | p95 | n |
|---|---|---|---|
| search | 2.8000 | 4.0000 | 30 |
| detail | 4.7000 | 7.6000 | 30 |
| similar_cold | 5.1000 | 6.8000 | 30 |
| similar_warm | 3.8000 | 4.9000 | 30 |
| home_cold | 41.4000 | 46.0000 | 30 |
| home_warm | 4.8000 | 9.8000 | 30 |
| collection_recs | 5.9000 | 7.7000 | 30 |
| chat_local | 6.8000 | 25.2000 | 30 |
| plan_preview | 167.6000 | 208.2000 | 30 |

Concurrency 5 (mixed reads): {'p50': 17.5, 'p95': 37.7, 'n': 60, 'errors': 0}

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
