## Ordering `Good Fit` above `Potential Fit`

256 rows carry one of the two labels, 130 of them `Good Fit`. Every ceiling below is cross-validated with the folds split by job description, so none of them is scoring the JD's identity.

| What is being measured | AUC |
|---|---:|
| the shipped weighted sum, as it ships | **0.411** |
| best reweighting of the same criterion scores | 0.451 |
| a supervised classifier on the raw text | 0.345 |

95% bootstrap interval on the shipped figure: **[0.340, 0.479]**.

The interval lies entirely **below** chance, so the ordering is real and it is **inverted**: the pipeline puts true `Potential Fit` above true `Good Fit` more often than not, and the sample is large enough to say so rather than to shrug. A score that is backwards carries signal; it is pointed the wrong way. The raw-text ceiling is 0.345, also below chance. A supervised classifier fitted on these very labels **does not transfer** between job descriptions -- what it learns on some is worse than a coin flip on others. That is a finding about how consistently the `Good Fit` / `Potential Fit` boundary was applied across this dataset, not about this agent, and it bounds what any pipeline could score here.

## Where each true class lands on the score

| True label | n | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|
| Good Fit | 130 | 0.000 | 0.048 | **0.325** | 0.591 | 1.000 |
| Potential Fit | 126 | 0.000 | 0.250 | **0.520** | 0.738 | 0.950 |
| No Fit | 244 | 0.000 | 0.000 | **0.230** | 0.500 | 1.000 |

The median true `Potential Fit` outscores the median true `Good Fit`. Between those two classes the ordering is not weak, it is inverted -- which is the same fact the AUC above reports, in the units a slide can show.

## What moving the cut points can and cannot buy

Grid search over both thresholds, fitted and scored on these same rows, so this is a ceiling and not a proposal.

- best macro-F1 reachable by re-cutting alone: **0.4132** at `good_fit_threshold = 0.79`, `potential_fit_threshold = 0.54`
- the shipped defaults (0.70 / 0.40) give **0.3939** on the same records

The whole calibration lever is worth **+0.0193 macro-F1**, and that figure is already overfitted to this split. The missing accuracy is not in where the score is cut.

## What a criterion score is, underneath

2868 criterion scores were written across these rows. **87.6%** of them (2513 scores) are exactly 0.0, 0.5 or 1.0: the model is not using the 0..1 range it was given, it is answering no, maybe or yes.

**1331** of them (46.4%) are 0.0, and **1192** of those (89.6%) reach the aggregator with no evidence attached at all -- no model-written quote and no hit from `expand_skill` either. Counted over the model's own answers in the cache rather than after the tools have added theirs, *every* 0.0 comes back without a quote and every non-zero score comes back with one. `SCORE_SYSTEM` asks for exactly this -- *"If nothing in the resume supports the criterion, score 0.0 and return no quotes"* -- so a CV that is silent on a criterion and a CV that plainly fails it reach `aggregate_scorecard` as the same number.

That conflation is real, but it is not the reason the ordering is lost. Rebuilding the weighted sum over only the criteria that actually found evidence is the obvious repair, and it does not work:

| Ordering | absence kept as 0.0 | absence dropped |
|---|---:|---:|
| `Good Fit` over `Potential Fit` | 0.444 | 0.403 |
| `Good Fit` over `No Fit` | 0.571 | 0.434 |

The contested pair stays at chance, the separable one gets worse, and 135 rows lose every criterion they had and drop out of the comparison entirely. The absent-as-zero signal is carrying most of what separates a fit of some kind from no fit, so removing it costs more than it returns.

## Can any run this project is allowed settle it

Sized against the 0.345 ceiling, because nothing in this pipeline should be expected to beat an upper bound fitted on the labels. Power is the chance a run of that size returns an interval that excludes the coin flip.

| Rows carrying one of the two labels | per class | power |
|---|---:|---:|
| 250 (test_500, if half its rows are contested) | 125 | 0.03 |
| 256 (dev_300, what is measured above) | 128 | 0.03 |
| 40000 (what 80% power needs) | 20000 | 0.03 |

`dev_300` answers this question with power **0.03**, which is why the interval above contains everything. `test_500` would carry roughly 250 contested rows if its label mix matches, for power **0.03** -- better, and still short of 80%. Settling it needs about **40000 contested rows**, which is more than either split has.

So the decision is not which repair to make to the scoring layer. It is that no run this project is allowed can show a repair worked, and a change that cannot be measured should not be made on the strength of a story about why it ought to help. The finding is the ceiling and the interval around it, and that is what belongs on the slide.
