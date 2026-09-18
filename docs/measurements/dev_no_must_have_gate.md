# agent / no_must_have_gate over 300 rows

**macro-F1: 0.4052** (300 scored, 0 errored)

| Label | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| Good Fit | 0.333 | 0.189 | 0.241 | 74 | 42 |
| Potential Fit | 0.320 | 0.427 | 0.366 | 75 | 100 |
| No Fit | 0.595 | 0.623 | 0.608 | 151 | 158 |

| Truth \ Predicted | Good Fit | Potential Fit | No Fit |
|---|---|---|---|
| Good Fit | 14 | 25 | 35 |
| Potential Fit | 14 | 32 | 29 |
| No Fit | 14 | 43 | 94 |

| Branch | Runs | Traffic |
|---|---:|---:|
| `guard -> quarantine` | 0 | 0.0% |
| `guard -> extract` | 300 | 100.0% |
| `extract -> repair` | 9 | 3.0% |
| `must_have_check -> reject_fast` | 0 | 0.0% |
| `must_have_check -> score_criteria` | 300 | 100.0% |
| `aggregate -> deep_review` | 64 | 21.3% |
| `aggregate -> decide` | 236 | 78.7% |

Cost: 1256880 tokens, 4189.6 per row, 109 live calls, 567 cache hits

Evidence coverage (E1): **0.424** of scored criteria carry a verbatim CV span.