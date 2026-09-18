# agent / shipped over 500 rows

**macro-F1: 0.3939** (500 scored, 0 errored)

| Label | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| Good Fit | 0.265 | 0.200 | 0.228 | 130 | 98 |
| Potential Fit | 0.351 | 0.317 | 0.333 | 126 | 114 |
| No Fit | 0.573 | 0.676 | 0.620 | 244 | 288 |

| Truth \ Predicted | Good Fit | Potential Fit | No Fit |
|---|---|---|---|
| Good Fit | 26 | 31 | 73 |
| Potential Fit | 36 | 40 | 50 |
| No Fit | 36 | 43 | 165 |

| Branch | Runs | Traffic |
|---|---:|---:|
| `guard -> quarantine` | 1 | 0.2% |
| `guard -> extract` | 499 | 99.8% |
| `extract -> repair` | 7 | 1.4% |
| `must_have_check -> reject_fast` | 121 | 24.2% |
| `must_have_check -> score_criteria` | 378 | 75.6% |
| `aggregate -> deep_review` | 77 | 15.4% |
| `aggregate -> decide` | 301 | 60.2% |

Cost: 1939927 tokens, 3879.9 per row, 816 live calls, 216 cache hits

Evidence coverage (E1): **0.486** of scored criteria carry a verbatim CV span.

## The must-have gate, judged against the labels

Rejected **121** rows before scoring. **43** of them were not truly `No Fit`, so the gate's precision is **0.645**.

| True label of a rejected row | Rows |
|---|---:|
| No Fit | 78 |
| Good Fit | 28 |
| Potential Fit | 15 |

| Blocking criterion | Rows |
|---|---:|
| `experience_years` | 48 |
| `skill_cplusplus` | 14 |
| `apache_spark_experience` | 11 |
| `cloud_technologies` | 9 |
| `oracle_rdbms` | 6 |
| `sql_experience` | 6 |
| `software_development_experience` | 5 |
| `years_of_experience` | 5 |
| `gaap_knowledge` | 4 |
| `programming_skills` | 4 |
| `tableau_skill` | 4 |
| `programming_language` | 3 |
| `snowflake_experience` | 3 |
| `sql_skill` | 3 |
| `database_knowledge` | 2 |
| `sap_experience` | 2 |
| `skill_python_sql` | 2 |
| `sql_expertise` | 2 |
| `data_engineering_experience` | 1 |
| `distributed_data_processing_experience` | 1 |
| `etl_processes` | 1 |
| `large_scale_data_systems_experience` | 1 |
| `microsoft_ssas` | 1 |
| `microsoft_ssis` | 1 |
| `microsoft_ssrs` | 1 |
| `ms_sql_experience` | 1 |
| `must_have_ansible` | 1 |
| `must_have_sap_experience` | 1 |
| `professional_experience` | 1 |
| `python_engineering` | 1 |
| `python_experience` | 1 |
| `python_programming` | 1 |
| `skill_erp_systems` | 1 |
| `skill_excel` | 1 |
| `skill_prega_rule_engine` | 1 |
| `sql_server` | 1 |
