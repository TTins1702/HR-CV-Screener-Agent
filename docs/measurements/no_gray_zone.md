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

## `shipped` against `no_gray_zone` over 300 rows

| | shipped | no_gray_zone | delta |
|---|---:|---:|---:|
| macro-F1 | 0.4155 | 0.4014 | -0.0140 |
| tokens | 1142849 | 1065293 | -77556 |
| errors | 0 | 0 | +0 |

Predictions that changed: **20** of 300.
