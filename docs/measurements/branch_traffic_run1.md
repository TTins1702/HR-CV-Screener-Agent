# Branch traffic over 300 real pairs

> **Superseded.** This is a log of a run made *before* the must-have gate was
> fixed, kept for history. Its `must_have_check -> reject_fast` figure of 32.0%
> is no longer the pipeline's behaviour: on the current records it is **18.3%**
> (55 of 300). Do not quote this file on a slide. The live numbers are in
> `dev_shipped.md`, regenerated with
> `python -m eval.report --records data/eval/after/agent__shipped.jsonl --out docs/measurements/dev_shipped.md --gate`.
>
> The token and tool-correction lines below cannot be regenerated from the
> record files, which is why this log is marked rather than rewritten.

| Branch | Runs | Traffic |
|---|---:|---:|
| `guard -> quarantine` | 0 | 0.0% |
| `guard -> extract` | 300 | 100.0% |
| `extract -> repair` | 9 | 3.0% |
| `must_have_check -> reject_fast` | 96 | 32.0% |
| `must_have_check -> score_criteria` | 204 | 68.0% |
| `aggregate -> deep_review` | 51 | 17.0% |
| `aggregate -> decide` | 153 | 51.0% |

Labels: {'Potential Fit': 79, 'No Fit': 191, 'Good Fit': 30}
Tokens: 1203546 total, 4011.8 per row, 567 live calls, 135 cache hits

calculate_experience corrected the model on 208 rows: median 3.75y, max 22.75y, 150 corrections of a year or more