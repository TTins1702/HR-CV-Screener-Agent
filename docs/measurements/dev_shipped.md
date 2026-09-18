# agent / shipped over 300 rows

**macro-F1: 0.4155** (300 scored, 0 errored)

| Label | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| Good Fit | 0.361 | 0.176 | 0.236 | 74 | 36 |
| Potential Fit | 0.337 | 0.400 | 0.366 | 75 | 89 |
| No Fit | 0.600 | 0.695 | 0.644 | 151 | 175 |

| Truth \ Predicted | Good Fit | Potential Fit | No Fit |
|---|---|---|---|
| Good Fit | 13 | 23 | 38 |
| Potential Fit | 13 | 30 | 32 |
| No Fit | 10 | 36 | 105 |

| Branch | Runs | Traffic |
|---|---:|---:|
| `guard -> quarantine` | 0 | 0.0% |
| `guard -> extract` | 300 | 100.0% |
| `extract -> repair` | 9 | 3.0% |
| `must_have_check -> reject_fast` | 55 | 18.3% |
| `must_have_check -> score_criteria` | 245 | 81.7% |
| `aggregate -> deep_review` | 56 | 18.7% |
| `aggregate -> decide` | 189 | 63.0% |

Cost: 1142849 tokens, 3809.5 per row, 0 live calls, 613 cache hits

Evidence coverage (E1): **0.439** of scored criteria carry a verbatim CV span.

## The must-have gate, judged against the labels

Rejected **55** rows before scoring. **20** of them were not truly `No Fit`, so the gate's precision is **0.636**.

| True label of a rejected row | Rows |
|---|---:|
| No Fit | 35 |
| Good Fit | 11 |
| Potential Fit | 9 |

| Blocking criterion | Rows |
|---|---:|
| `experience_years` | 17 |
| `accounting_experience` | 3 |
| `bash_scripting` | 3 |
| `automotive_experience` | 2 |
| `cc_cpp_experience` | 2 |
| `devops_experience` | 2 |
| `experience_years_salesforce` | 2 |
| `experience_years_software_engineering` | 2 |
| `gaap_knowledge` | 2 |
| `salesforce_experience` | 2 |
| `sap_experience` | 2 |
| `skill_jira_tools` | 2 |
| `skill_tableau` | 2 |
| `asc_606_knowledge` | 1 |
| `cloud_data_solutions` | 1 |
| `computer_science_fundamentals` | 1 |
| `core_java_selenium` | 1 |
| `experience_years_b2b_saas` | 1 |
| `experience_years_leadership` | 1 |
| `experience_years_pern_stack` | 1 |
| `experience_years_web_apps` | 1 |
| `experience_years_web_development` | 1 |
| `gaap_proficiency` | 1 |
| `gcp_experience` | 1 |
| `graphics_api_experience` | 1 |
| `graphics_programming_experience` | 1 |
| `intacct_proficiency` | 1 |
| `java_skill` | 1 |
| `ms_excel_skills` | 1 |
| `must_have_apache_spark_expertise` | 1 |
| `must_have_experience_years` | 1 |
| `python_skill` | 1 |
| `quickbooks_proficiency` | 1 |
| `schematics_knowledge` | 1 |
| `skill_sql` | 1 |
| `skill_yardi` | 1 |
| `sql_proficiency` | 1 |
| `sql_queries` | 1 |
| `strong_excel_skills` | 1 |
