# `extraction_confidence`: keep it, report it, never route on it

Measured over the 1030 unique extraction answers in `data/cache/llm_cache.jsonl`. Regenerate with `python -m eval.confidence_probe`.

## It is a four-value self-report, not a confidence

| Value | Answers | Share |
|---|---:|---:|
| 0.00 | 245 | 23.8% |
| 0.90 | 689 | 66.9% |
| 0.95 | 77 | 7.5% |
| 1.00 | 19 | 1.8% |

The extract node's docstring used to say it *"never dropped below 0.90"*. It does, on **23.8%** of extractions, and it takes only four distinct values across the whole cache. A number that moves between four points is a label wearing a decimal point.

## It is informative, and redundant

The same docstring called it uninformative. That was also wrong. A zero tracks whether a total experience figure came out at all:

| | total_experience_years missing | present |
|---|---:|---:|
| confidence == 0.00 | 221 | 24 |
| confidence > 0.00 | 125 | 660 |

As a detector of a missing years field it scores precision **0.902** and recall **0.639**.

That is the case against routing on it, and it is a stronger case than the one it replaces. The field is **redundant**: `total_experience_years is None` answers the same question exactly, deterministically, and without asking a model to grade its own work. A branch built on the self-report would be a worse copy of a branch available for free.

## Why it is not simply deleted

`extraction_confidence` is a field of `RawExtraction`, and the cache key covers the response schema. Removing it changes every extraction key on disk, so the next run is cold and every measurement already published was taken against a different contract. Carrying an unused field is the cheaper of the two mistakes.

So: it stays in the contract, it goes to the recruiter-facing view labelled for what it is rather than as a confidence, and `test_nothing_in_the_graph_routes_on_extraction_confidence` keeps it out of the control flow.
