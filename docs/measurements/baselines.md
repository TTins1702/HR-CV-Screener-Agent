# dev_300: the agent against the three spec section 7 baselines

Measured 2026-09-17, all four systems over the same 300 rows, scored by the same
functions in `eval/metrics.py`. `test_500` has not been touched.

| System | macro-F1 | Good Fit recall | Good Fit F1 | No Fit F1 | Tokens | Evidence (E1) |
|---|---:|---:|---:|---:|---:|---:|
| `agent` | 0.4155 | 0.176 | 0.236 | 0.644 | 1142849 | 0.439 |
| `baseline_naive` | 0.3742 | 0.041 | 0.066 | 0.700 | 521102 | 0.000 |
| `baseline_rubric` | 0.3750 | 0.041 | 0.068 | 0.699 | 453321 | 0.000 |
| `baseline_tfidf` | 0.3673 | 0.419 | 0.356 | 0.494 | 0 | 0.000 |

## The agent wins, by less than it looks

0.4155 against 0.3750 for the strongest baseline is a gap of **0.04 macro-F1** for
**2.5x the tokens** of the rubric-in-prompt call. Spec section 7 anticipated exactly
this and says what to do about it: if the strong baseline is close, the agent's case
rests on evidence linkage, injection defence, broken-CV handling and cost. The last
column is the first of those, and it is not close -- **0.439 against 0.000**. No
single-call baseline can point at the sentence in the CV that produced its answer,
because it never produced one.

## TF-IDF finds Good Fit candidates better than the agent does

Bag-of-words cosine, no model, zero tokens, gets **Good Fit recall 0.419** against the
agent's 0.176 and a better Good Fit F1 (0.356 against 0.236). The agent's whole margin
comes from `No Fit`. Reported here rather than buried: a screening tool whose headline
job is surfacing strong candidates is currently beaten at that job by a technique from
1972.

## Ordering Good Fit above Potential Fit

AUC, over the systems that produce a continuous score at all. 0.5 is a coin flip.

| System | AUC Good Fit over Potential Fit |
|---|---:|
| `agent` | 0.482 |
| `baseline_tfidf` | 0.526 |

Both are at chance, and `scoring_layer.md` says how little room there is above them.
Re-cutting the thresholds alone reaches 0.4365 macro-F1 against the shipped 0.4155, and
a *supervised* classifier fitted on the raw text -- cross-validated with the folds split
by job description, so it is not scoring the JD's identity -- reaches only 0.577 AUC on
the same pair. That last number is the one to quote: it is an upper bound built with the
labels the agent never sees, and it is still barely above a coin flip.

The honest caveat is that all of this rests on 149 rows. The bootstrap interval on the
agent's 0.482 is [0.386, 0.575], which contains chance *and* contains the 0.577 ceiling.
So the slide can say the boundary is close to unrecoverable from a CV and a JD, and that
this is a fact about the task rather than about this agent -- but it cannot say the
scoring layer is or is not throwing signal away, because this split cannot tell.

## One baseline had to be rewritten before it could be used

The first rubric-in-prompt prompt ended with "A candidate missing a criterion marked
must-have cannot be a Good Fit". Nearly every derived rubric carries a must-have and
the model could not verify them from the CV, so it answered `No Fit` on **297 of 300**
rows and scored macro-F1 0.2405. That number was a measurement of the prompt, not of
the baseline, and reporting it would have handed the agent a win it did not earn.
Rewritten, the same baseline scores 0.3750. `eval/report.py` now flags any run whose
predictions are 95% one label, so the next collapse is visible rather than plausible.
