# System evaluation - synthetic data

In-process FastAPI (TestClient) on a copy of the activated database; single machine (see manifest.json). `*_cold` clears the recommendation cache before each call.

| endpoint | p50 | p95 | n |
|---|---|---|---|
| search | 2.2000 | 3.4000 | 30 |
| detail | 3.4000 | 5.2000 | 30 |
| similar_cold | 3.7000 | 6.1000 | 30 |
| similar_warm | 2.6000 | 3.6000 | 30 |
| home_cold | 46.3000 | 49.6000 | 30 |
| home_warm | 3.5000 | 4.7000 | 30 |
| collection_recs | 4.2000 | 5.7000 | 30 |
| chat_local | 4.7000 | 19.5000 | 30 |
| plan_preview | 162.4000 | 192.9000 | 30 |

Concurrency 5 (mixed reads): {'p50': 11.4, 'p95': 19.0, 'n': 60, 'errors': 0}

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
