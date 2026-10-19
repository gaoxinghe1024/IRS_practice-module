# Explicit-rating diagnostic (MAE / RMSE) - real data

Dataset `real-034ebd6a74`, explicit protocol (per-user random 80/10/10). Test pairs scored: 239,937 from 20,000 users. All methods predict the same pairs on the 1-10 scale.

| method | mae | rmse | mae_on_icf_supported |
|---|---|---|---|
| global_mean | 1.3036 | 1.6444 | 1.3052 |
| user_mean | 1.0996 | 1.4354 | 1.0986 |
| item_mean | 1.1590 | 1.5032 | 1.1417 |
| icf_with_fallback | 0.9656 | 1.3474 | 0.9369 |
| icf_served_with_fallback | 1.0337 | 1.3593 | 1.0191 |
| mf | 0.8719 | 1.1683 | 0.8481 |
| ncf_explicit | 0.9198 | 1.2411 | 0.8999 |
| icf_supported_only | 0.9369 | 1.3172 | 0.9369 |

ICF has neighbour support for 82.9% of test pairs; `icf_with_fallback` uses the user's train mean elsewhere and `icf_supported_only` restricts to supported pairs (not directly comparable with full-set rows).

Training times: icf_fit_s=58.6s, mf_fit_s=115.7s, ncf_fit_s=306.3s

MAE/RMSE measure rating prediction, not ranking quality; the homepage experiment reports ranking metrics on the separate ranking protocol.

- note: 8 test pairs dropped: user or item absent from explicit-train (cold pairs)
- note: icf = exact formula; icf_served = served configuration (damping 20.0, user-mean prior weight 5.0)
