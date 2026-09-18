# Branch traffic over 25 real pairs

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
| `guard -> extract` | 25 | 100.0% |
| `extract -> repair` | 1 | 4.0% |
| `must_have_check -> reject_fast` | 8 | 32.0% |
| `must_have_check -> score_criteria` | 17 | 68.0% |
| `aggregate -> deep_review` | 5 | 20.0% |
| `aggregate -> decide` | 12 | 48.0% |

Labels: {'Potential Fit': 8, 'No Fit': 15, 'Good Fit': 2}
Tokens: 84173 total, 3366.9 per row, 0 live calls, 48 cache hits

calculate_experience corrected the model on 19 rows: median 1.5y, max 15.08y, 11 corrections of a year or more