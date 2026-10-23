# Explicit-rating diagnostic (MAE / RMSE) - synthetic data

Dataset `synthetic-7a7d5d6a74`, explicit protocol (per-user random 80/10/10). Test pairs scored: 1,647 from 608 users. All methods predict the same pairs on the 1-10 scale.

| method | mae | rmse | mae_on_icf_supported |
|---|---|---|---|
| global_mean | 1.2884 | 1.6166 | 1.2890 |
| user_mean | 1.2117 | 1.4810 | 1.2115 |
| item_mean | 1.2577 | 1.5633 | 1.2584 |
| icf_with_fallback | 1.1017 | 1.3852 | 1.1011 |
| icf_served_with_fallback | 1.1492 | 1.4082 | 1.1488 |
| mf | 1.0509 | 1.3175 | 1.0499 |
| ncf_explicit | 1.0471 | 1.3071 | 1.0464 |
| icf_supported_only | 1.1011 | 1.3843 | 1.1011 |

ICF has neighbour support for 99.6% of test pairs; `icf_with_fallback` uses the user's train mean elsewhere and `icf_supported_only` restricts to supported pairs (not directly comparable with full-set rows).

Training times: icf_fit_s=0.0s, mf_fit_s=0.1s, ncf_fit_s=1.7s

MAE/RMSE measure rating prediction, not ranking quality; the homepage experiment reports ranking metrics on the separate ranking protocol.

- note: 0 test pairs dropped: user or item absent from explicit-train (cold pairs)
- note: icf = exact formula; icf_served = served configuration (damping 5.0, user-mean prior weight 10.0)
