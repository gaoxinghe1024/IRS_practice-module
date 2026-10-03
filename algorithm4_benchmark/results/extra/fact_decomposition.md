# table_fact_decomposition — 8 factoid questions x 6 models (48 rows)

| qid | field | expect | model | std_in_top3 | value_in_prompt | N_ok | R_ok | flip | R_refused | flip_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| F01 | studios | Brain's Base | Hy3 | False | False | True | False | True | False | context_missing_field |
| F01 | studios | Brain's Base | Deepseek-V4.1-Flash | False | False | True | False | True | False | context_missing_field |
| F01 | studios | Brain's Base | GLM-5.3 | False | False | True | False | True | True | context_missing_field |
| F01 | studios | Brain's Base | GLM-5.3-Flash | False | False | True | False | True | True | context_missing_field |
| F01 | studios | Brain's Base | MiniMax-M3 | False | False | True | False | True | True | context_missing_field |
| F01 | studios | Brain's Base | Kimi-K3 | False | False | True | False | True | True | context_missing_field |
| F02 | aired | 2022 | Hy3 | False | False | True | False | True | True | context_missing_field |
| F02 | aired | 2022 | Deepseek-V4.1-Flash | False | False | True | False | True | True | context_missing_field |
| F02 | aired | 2022 | GLM-5.3 | False | False | True | False | True | True | context_missing_field |
| F02 | aired | 2022 | GLM-5.3-Flash | False | False | True | False | True | True | context_missing_field |
| F02 | aired | 2022 | MiniMax-M3 | False | False | True | False | True | True | context_missing_field |
| F02 | aired | 2022 | Kimi-K3 | False | False | True | False | True | True | context_missing_field |
| F03 | an_type | Movie | Hy3 | True | False | True | True | False | False |  |
| F03 | an_type | Movie | Deepseek-V4.1-Flash | True | False | True | True | False | False |  |
| F03 | an_type | Movie | GLM-5.3 | True | False | True | False | True | False | context_missing_field |
| F03 | an_type | Movie | GLM-5.3-Flash | True | False | True | True | False | False |  |
| F03 | an_type | Movie | MiniMax-M3 | True | False | True | True | False | False |  |
| F03 | an_type | Movie | Kimi-K3 | True | False | True | True | False | False |  |
| F04 | episodes | 26 | Hy3 | True | True | True | True | False | False |  |
| F04 | episodes | 26 | Deepseek-V4.1-Flash | True | True | True | True | False | False |  |
| F04 | episodes | 26 | GLM-5.3 | True | True | True | True | False | False |  |
| F04 | episodes | 26 | GLM-5.3-Flash | True | True | True | True | False | False |  |
| F04 | episodes | 26 | MiniMax-M3 | True | True | True | True | False | False |  |
| F04 | episodes | 26 | Kimi-K3 | True | True | True | True | False | False |  |
| F05 | studios | MAPPA | Hy3 | False | False | True | False | True | True | context_missing_field |
| F05 | studios | MAPPA | Deepseek-V4.1-Flash | False | False | True | False | True | True | context_missing_field |
| F05 | studios | MAPPA | GLM-5.3 | False | False | True | False | True | True | context_missing_field |
| F05 | studios | MAPPA | GLM-5.3-Flash | False | False | True | False | True | True | context_missing_field |
| F05 | studios | MAPPA | MiniMax-M3 | False | False | True | False | True | True | context_missing_field |
| F05 | studios | MAPPA | Kimi-K3 | False | False | True | False | True | True | context_missing_field |
| F06 | episodes | 148 | Hy3 | True | True | True | True | False | False |  |
| F06 | episodes | 148 | Deepseek-V4.1-Flash | True | True | True | True | False | False |  |
| F06 | episodes | 148 | GLM-5.3 | True | True | True | True | False | False |  |
| F06 | episodes | 148 | GLM-5.3-Flash | True | True | True | True | False | False |  |
| F06 | episodes | 148 | MiniMax-M3 | True | True | True | True | False | False |  |
| F06 | episodes | 148 | Kimi-K3 | True | True | True | True | False | False |  |
| F07 | aired | 1995 | Hy3 | False | False | True | False | True | True | context_missing_field |
| F07 | aired | 1995 | Deepseek-V4.1-Flash | False | False | True | False | True | True | context_missing_field |
| F07 | aired | 1995 | GLM-5.3 | False | False | True | False | True | True | context_missing_field |
| F07 | aired | 1995 | GLM-5.3-Flash | False | False | True | False | True | False | context_missing_field |
| F07 | aired | 1995 | MiniMax-M3 | False | False | True | False | True | True | context_missing_field |
| F07 | aired | 1995 | Kimi-K3 | False | False | True | True | False | True |  |
| F08 | episodes | 201 | Hy3 | False | True | True | False | True | True | work_not_retrieved |
| F08 | episodes | 201 | Deepseek-V4.1-Flash | False | True | True | False | True | True | work_not_retrieved |
| F08 | episodes | 201 | GLM-5.3 | False | True | True | False | True | True | work_not_retrieved |
| F08 | episodes | 201 | GLM-5.3-Flash | False | True | True | False | True | True | work_not_retrieved |
| F08 | episodes | 201 | MiniMax-M3 | False | True | True | False | True | True | work_not_retrieved |
| F08 | episodes | 201 | Kimi-K3 | False | True | True | False | True | True | work_not_retrieved |