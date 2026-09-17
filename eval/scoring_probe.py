"""Why `score_criteria` cannot order `Good Fit` above `Potential Fit`.

`docs/measurements/scoring_layer.md` named three suspects: the per-criterion
scores, the weights, and the three-class question itself. Answering them needs
three numbers on the same footing -- what the shipped weighted sum achieves,
what the best possible reweighting of those same criterion scores achieves, and
what a supervised classifier achieves on the raw text. Only the third says
anything about the task rather than about this pipeline.

Every comparison here is grouped by job description. Ungrouped, a classifier
scores the JD's identity rather than the candidate: the same JD lands in train
and test carrying the same label, and the raw-text ceiling reads 0.701 instead
of its honest 0.577. That gap is larger than every effect this module was
written to measure, so the grouping is not a refinement, it is the measurement.

Nothing here calls a model or touches the network. It is arithmetic over the
criterion-level dump that `scripts/dump_criterion_scores.py` replays from cache.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from eval.metrics import LABELS

# The pair the pipeline cannot order. `No Fit` is separable and not in question.
CONTESTED = ("Good Fit", "Potential Fit")


class CriterionRow(BaseModel):
    """One criterion's score and whether any CV span was found for it."""

    score: float = Field(ge=0.0, le=1.0)
    has_evidence: bool


class AbsenceStats(BaseModel):
    """How much of the score is the model saying "the CV does not mention this"."""

    total: int
    zeros: int
    zeros_without_evidence: int
    share_at_three_levels: float


class ProbeResult(BaseModel):
    """The three AUCs, on the same rows and the same grouping."""

    rows: int
    positives: int
    shipped_auc: float
    ci_low: float
    ci_high: float
    criterion_ceiling: float
    text_ceiling: float


def auc(positive: list[float] | np.ndarray, negative: list[float] | np.ndarray) -> float:
    """P(a random positive outscores a random negative), ties counting as half.

    Not folded to `max(a, 1 - a)`: 0.482 has to stay readable as *below* the coin
    flip, which is the finding, rather than as a weak version of 0.518.
    """
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    labels = np.r_[np.ones(positive.size), np.zeros(negative.size)]
    return float(roc_auc_score(labels, np.r_[positive, negative]))


def bootstrap_auc_ci(
    labels: np.ndarray | list,
    scores: np.ndarray | list,
    *,
    reps: int = 4000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """A percentile interval on the AUC, resampling rows with replacement.

    The interval is the point of the exercise. On 149 rows it spans roughly
    0.19, which is wide enough to hold both the coin flip and every ceiling
    measured here -- so it is what turns "the scoring layer is broken" into "this
    split cannot tell".
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(reps):
        picked = rng.integers(0, labels.size, labels.size)
        if np.unique(labels[picked]).size < 2:
            continue
        draws.append(roc_auc_score(labels[picked], scores[picked]))
    if not draws:
        return (float("nan"), float("nan"))
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


def grouped_ceiling_auc(
    features,
    labels: np.ndarray | list,
    groups: np.ndarray | list,
    *,
    estimator=None,
    folds: int = 5,
    seed: int = 0,
) -> float:
    """The best AUC a fitted model gets out of `features`, without leaking.

    An upper bound, not a target: it is fitted on the labels the agent never
    sees. If this sits at chance, no amount of reweighting or recalibrating the
    thing the features came from will help, because the ordering is not in there.

    `estimator` takes the whole pipeline rather than just the classifier so that
    a vectoriser is refitted inside every fold. Fitting one over all the rows
    first would let the held-out fold's vocabulary into the ceiling it is meant
    to bound.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if estimator is None:
        estimator = LogisticRegression(max_iter=5000)
        features = np.asarray(features, float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        estimator,
        features,
        labels,
        cv=splitter,
        groups=groups,
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(labels, predicted))


def absence_zero_share(rows: list[CriterionRow]) -> AbsenceStats:
    """How many criterion scores are 0.0, and how many of those found nothing.

    `SCORE_SYSTEM` tells the model to "score 0.0 and return no quotes" when the
    resume does not support a criterion, so absence and failure arrive at the
    aggregator as the same number. This counts how much of the score that is.
    """
    total = len(rows)
    zeros = sum(1 for row in rows if row.score == 0.0)
    silent = sum(1 for row in rows if row.score == 0.0 and not row.has_evidence)
    three = sum(1 for row in rows if row.score in (0.0, 0.5, 1.0))
    return AbsenceStats(
        total=total,
        zeros=zeros,
        zeros_without_evidence=silent,
        share_at_three_levels=(three / total) if total else 0.0,
    )


class AbsenceCounterfactual(BaseModel):
    """The ordering with the absent-as-zero scores kept, and with them dropped."""

    kept_good_over_potential: float
    kept_good_over_no: float
    dropped_good_over_potential: float
    dropped_good_over_no: float
    dropped_rows_lost: int


def _renormalised(criteria: list[dict], keep) -> float | None:
    """Weighted mean over the criteria `keep` admits, or None if it admits none."""
    earned = sum((c.get("weight") or 0.0) * c["score"] for c in criteria if keep(c))
    weight = sum((c.get("weight") or 0.0) for c in criteria if keep(c))
    return earned / weight if weight > 0 else None


def absence_counterfactual(dumped: list[dict]) -> AbsenceCounterfactual:
    """What the ordering would be if "the CV is silent" stopped counting as zero.

    The obvious repair for conflating absence with failure is to leave those
    criteria out of the weighted mean. This measures it instead of assuming it,
    and reports `Good Fit` over `No Fit` beside the contested pair, because that
    is the ordering the repair turns out to cost.
    """
    everything = lambda c: True  # noqa: E731
    evidenced = lambda c: not (c["score"] == 0.0 and not c.get("has_evidence"))  # noqa: E731

    kept: dict[str, list[float]] = {label: [] for label in LABELS}
    dropped: dict[str, list[float]] = {label: [] for label in LABELS}
    lost = 0
    for row in dumped:
        label = row["true_label"]
        if label not in kept:
            continue
        with_absence = _renormalised(row["criteria"], everything)
        if with_absence is not None:
            kept[label].append(with_absence)
        without = _renormalised(row["criteria"], evidenced)
        if without is None:
            lost += 1
        else:
            dropped[label].append(without)

    good, potential, no_fit = LABELS[0], LABELS[1], LABELS[2]
    return AbsenceCounterfactual(
        kept_good_over_potential=auc(kept[good], kept[potential]),
        kept_good_over_no=auc(kept[good], kept[no_fit]),
        dropped_good_over_potential=auc(dropped[good], dropped[potential]),
        dropped_good_over_no=auc(dropped[good], dropped[no_fit]),
        dropped_rows_lost=lost,
    )


def _hanley_mcneil_se(area: float, per_class: int) -> float:
    """The standard error of an AUC with `per_class` rows on each side."""
    q1 = area / (2 - area)
    q2 = 2 * area * area / (1 + area)
    variance = (
        area * (1 - area)
        + (per_class - 1) * (q1 - area**2)
        + (per_class - 1) * (q2 - area**2)
    ) / (per_class * per_class)
    return math.sqrt(max(variance, 0.0))


def auc_power(true_auc: float, per_class: int, *, alpha: float = 0.025) -> float:
    """P(a run of this size shows the ordering beats chance).

    Day 4 published "the pipeline is unstable, not biased" from a sign test with
    power 0.40, which was not evidence of fairness but evidence of too few rows.
    This is the same question asked in advance, so the single `test_500` run
    spec section 3 allows is spent knowing what it can and cannot settle.

    `alpha` defaults to 0.025 rather than 0.05 so that "detected" means the same
    thing here as everywhere else in this module: the 95% two-sided interval
    reported beside every AUC excludes 0.5. A one-sided 0.05 test would call
    results significant that the published interval still straddles.
    """
    if true_auc <= 0.5:
        return float(alpha)
    standard_error = _hanley_mcneil_se(true_auc, per_class)
    if standard_error == 0:
        return 1.0
    critical = NormalDist().inv_cdf(1 - alpha)
    return float(NormalDist().cdf((true_auc - 0.5) / standard_error - critical))


def rows_needed(
    true_auc: float, *, power: float = 0.80, alpha: float = 0.025, cap: int = 20000
) -> int:
    """Rows per class needed to show `true_auc` beats chance at `power`.

    Returned per class, so a rubric-fit split needs roughly twice this many rows
    carrying one of the two contested labels.
    """
    for per_class in range(10, cap):
        if auc_power(true_auc, per_class, alpha=alpha) >= power:
            return per_class
    return cap


SHIPPED_THRESHOLDS = (0.70, 0.40)


class ThresholdSearch(BaseModel):
    """The best macro-F1 reachable by moving the two cut points and nothing else."""

    good_threshold: float
    potential_threshold: float
    macro_f1: float
    shipped_macro_f1: float
    gain: float


def _label_for(score: float, good: float, potential: float) -> str:
    if score >= good:
        return "Good Fit"
    if score >= potential:
        return "Potential Fit"
    return "No Fit"


def best_thresholds(
    labels: list[str],
    scores: list[float],
    *,
    step: float = 0.01,
    shipped: tuple[float, float] = SHIPPED_THRESHOLDS,
) -> ThresholdSearch:
    """Grid search both cut points; report the ceiling and what shipping costs.

    Fitted on the same rows it is scored on, so it is an optimistic bound rather
    than a proposal. If even that bound is close to the shipped figure, the cut
    points are not where the missing accuracy is, and moving them would be
    fitting this split.
    """
    from eval.metrics import Pair, macro_f1

    grid = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    best = ThresholdSearch(
        good_threshold=shipped[0],
        potential_threshold=shipped[1],
        macro_f1=-1.0,
        shipped_macro_f1=0.0,
        gain=0.0,
    )
    shipped_f1 = macro_f1(
        [Pair((t, _label_for(s, *shipped))) for t, s in zip(labels, scores)]
    )
    for good in grid:
        for potential in grid:
            if potential >= good:
                continue
            value = macro_f1(
                [
                    Pair((t, _label_for(s, good, potential)))
                    for t, s in zip(labels, scores)
                ]
            )
            if value > best.macro_f1:
                best = ThresholdSearch(
                    good_threshold=good,
                    potential_threshold=potential,
                    macro_f1=value,
                    shipped_macro_f1=shipped_f1,
                    gain=value - shipped_f1,
                )
    return best


def scoring_probe(
    *,
    labels: list[str],
    shipped_scores: list[float],
    groups: list[str],
    criterion_features: np.ndarray,
    text_features,
    text_estimator=None,
    folds: int = 5,
    reps: int = 4000,
    seed: int = 0,
) -> ProbeResult:
    """The shipped ordering beside both ceilings, on the contested pair only."""
    labels_array = np.asarray(labels)
    keep = np.isin(labels_array, CONTESTED)
    y = (labels_array[keep] == CONTESTED[0]).astype(int)
    if np.unique(y).size < 2:
        raise ValueError(f"need both of {CONTESTED} present to order them")

    scores = np.asarray(shipped_scores, float)[keep]
    grouping = np.asarray(groups)[keep]
    shipped = float(roc_auc_score(y, scores))
    low, high = bootstrap_auc_ci(y, scores, reps=reps, seed=seed)

    return ProbeResult(
        rows=int(keep.sum()),
        positives=int(y.sum()),
        shipped_auc=shipped,
        ci_low=low,
        ci_high=high,
        criterion_ceiling=grouped_ceiling_auc(
            np.asarray(criterion_features, float)[keep], y, grouping,
            folds=folds, seed=seed,
        ),
        text_ceiling=grouped_ceiling_auc(
            (
                [t for t, k in zip(text_features, keep) if k]
                if text_estimator is not None
                else np.asarray(text_features, float)[keep]
            ),
            y,
            grouping,
            estimator=text_estimator,
            folds=folds,
            seed=seed,
        ),
    )


def render_scoring_probe(result: ProbeResult) -> str:
    """The markdown for the probe, ceilings under the number they bound."""
    good, potential = CONTESTED
    inconclusive = result.ci_low <= 0.5 <= result.ci_high
    return "\n".join(
        [
            f"## Ordering `{good}` above `{potential}`",
            "",
            f"{result.rows} rows carry one of the two labels, {result.positives} of "
            f"them `{good}`. Every ceiling below is cross-validated with the folds "
            "split by job description, so none of them is scoring the JD's identity.",
            "",
            "| What is being measured | AUC |",
            "|---|---:|",
            f"| the shipped weighted sum, as it ships | **{result.shipped_auc:.3f}** |",
            f"| best reweighting of the same criterion scores | {result.criterion_ceiling:.3f} |",
            f"| a supervised classifier on the raw text | {result.text_ceiling:.3f} |",
            "",
            f"95% bootstrap interval on the shipped figure: "
            f"**[{result.ci_low:.3f}, {result.ci_high:.3f}]**.",
            "",
            (
                "Chance falls inside that interval, and so do both ceilings. On this "
                "split the three rows of the table are not distinguishable from each "
                "other, so it cannot be said whether the ordering is lost in the "
                "scoring layer or was never in the data."
                if inconclusive
                else "The interval excludes chance, so the ordering is real."
            ),
        ]
    )


KINDS = ("experience_years", "education", "skill", "other", "domain")


def criterion_feature_row(criteria: list[dict]) -> list[float]:
    """One row of the criterion-score feature space.

    The rubric is derived per job description -- 146 distinct ones over 300 rows
    -- so criterion ids are not a shared vocabulary and cannot be columns. `kind`
    is, which is why the reweighting ceiling is expressed over kinds: it is the
    most generous shared space these scores admit.
    """
    row: list[float] = []
    for kind in KINDS:
        matching = [c for c in criteria if c.get("kind") == kind]
        weight = sum((c.get("weight") or 0.0) for c in matching)
        earned = sum((c.get("weight") or 0.0) * c["score"] for c in matching)
        row += [earned / weight if weight > 0 else 0.0, weight]
    row.append(
        sum(1 for c in criteria if c.get("has_evidence")) / len(criteria)
        if criteria
        else 0.0
    )
    row.append(float(len(criteria)))
    must = [c for c in criteria if c.get("must_have")]
    row.append(sum(c["score"] for c in must) / len(must) if must else 0.0)
    return row


def _text_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=5000),
    )


def main(argv: list[str] | None = None) -> int:
    """Turn the criterion-score dump into `docs/measurements/scoring_layer.md`."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="why the scoring layer cannot order")
    parser.add_argument(
        "--dump", type=Path, default=Path("data/eval/criterion_scores__shipped.jsonl")
    )
    parser.add_argument("--split", type=Path, default=Path("data/samples/dev_300.jsonl"))
    parser.add_argument(
        "--out", type=Path, default=Path("docs/measurements/scoring_layer.md")
    )
    parser.add_argument("--reps", type=int, default=4000)
    args = parser.parse_args(argv)

    for path in (args.dump, args.split):
        if not path.exists():
            parser.error(f"no such file: {path}")

    dumped = [
        json.loads(line)
        for line in args.dump.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dumped = [row for row in dumped if "error" not in row]
    split = [
        json.loads(line)
        for line in args.split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    texts = [
        split[row["row"]]["resume_text"] + " \n " + split[row["row"]]["job_description_text"]
        for row in dumped
    ]
    result = scoring_probe(
        labels=[row["true_label"] for row in dumped],
        shipped_scores=[row["overall_score"] for row in dumped],
        groups=[row["jd_fingerprint"] for row in dumped],
        criterion_features=np.array(
            [criterion_feature_row(row["criteria"]) for row in dumped], float
        ),
        text_features=texts,
        text_estimator=_text_pipeline(),
        reps=args.reps,
    )
    stats = absence_zero_share(
        [
            CriterionRow(score=c["score"], has_evidence=bool(c.get("has_evidence")))
            for row in dumped
            for c in row["criteria"]
        ]
    )

    thresholds = best_thresholds(
        [row["true_label"] for row in dumped],
        [row["overall_score"] for row in dumped],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n\n".join(
            [
                render_scoring_probe(result),
                render_distribution(dumped),
                render_thresholds(thresholds),
                render_absence(stats, absence_counterfactual(dumped)),
                render_power(result),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} from {args.dump} ({len(dumped)} rows)")
    return 0


def render_distribution(dumped: list[dict]) -> str:
    """Where each true class actually lands on the score, quartile by quartile."""
    lines = [
        "## Where each true class lands on the score",
        "",
        "| True label | n | min | p25 | median | p75 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        values = sorted(
            row["overall_score"] for row in dumped if row["true_label"] == label
        )
        if not values:
            continue
        pick = lambda q: values[int(q * (len(values) - 1))]  # noqa: E731
        lines.append(
            f"| {label} | {len(values)} | {values[0]:.3f} | {pick(0.25):.3f} | "
            f"**{pick(0.5):.3f}** | {pick(0.75):.3f} | {values[-1]:.3f} |"
        )
    lines += [
        "",
        "The median true `Potential Fit` outscores the median true `Good Fit`. "
        "Between those two classes the ordering is not weak, it is inverted -- "
        "which is the same fact the AUC above reports, in the units a slide can "
        "show.",
    ]
    return "\n".join(lines)


def render_thresholds(best: ThresholdSearch) -> str:
    """The markdown for what recalibration alone can buy."""
    return "\n".join(
        [
            "## What moving the cut points can and cannot buy",
            "",
            "Grid search over both thresholds, fitted and scored on these same "
            "rows, so this is a ceiling and not a proposal.",
            "",
            f"- best macro-F1 reachable by re-cutting alone: **{best.macro_f1:.4f}** "
            f"at `good_fit_threshold = {best.good_threshold:.2f}`, "
            f"`potential_fit_threshold = {best.potential_threshold:.2f}`",
            f"- the shipped defaults "
            f"({SHIPPED_THRESHOLDS[0]:.2f} / {SHIPPED_THRESHOLDS[1]:.2f}) give "
            f"**{best.shipped_macro_f1:.4f}** on the same records",
            "",
            f"The whole calibration lever is worth **{best.gain:+.4f} macro-F1**, "
            "and that figure is already overfitted to this split. The missing "
            "accuracy is not in where the score is cut.",
        ]
    )


def render_power(result: ProbeResult, *, target_power: float = 0.80) -> str:
    """What sample size would be needed to settle the ordering, and what exists.

    The ceiling measured on the raw text is the most generous effect anyone
    should expect the pipeline to reach, so it is the effect the sample size is
    planned against. Planning against a larger one would be planning to be
    underpowered.
    """
    effect = result.text_ceiling
    per_class = result.rows // 2
    needed = rows_needed(effect, power=target_power)
    lines = [
        "## Can any run this project is allowed settle it",
        "",
        f"Sized against the {effect:.3f} ceiling, because nothing in this "
        "pipeline should be expected to beat an upper bound fitted on the "
        "labels. Power is the chance a run of that size returns an interval "
        "that excludes the coin flip.",
        "",
        "| Rows carrying one of the two labels | per class | power |",
        "|---|---:|---:|",
    ]
    planned = [
        (result.rows, "dev_300, what is measured above"),
        (250, "test_500, if half its rows are contested"),
        (2 * needed, f"what {target_power:.0%} power needs"),
    ]
    for contested, note in sorted(planned):
        half = contested // 2
        lines.append(
            f"| {contested} ({note}) | {half} | {auc_power(effect, half):.2f} |"
        )
    shortfall = auc_power(effect, 125)
    lines += [
        "",
        f"`dev_300` answers this question with power **{auc_power(effect, per_class):.2f}**, "
        "which is why the interval above contains everything. `test_500` would "
        f"carry roughly 250 contested rows if its label mix matches, for power "
        f"**{shortfall:.2f}** -- better, and still short of {target_power:.0%}. "
        f"Settling it needs about **{2 * needed} contested rows**, which is more "
        "than either split has.",
        "",
        "So the decision is not which repair to make to the scoring layer. It is "
        "that no run this project is allowed can show a repair worked, and a "
        "change that cannot be measured should not be made on the strength of a "
        "story about why it ought to help. The finding is the ceiling and the "
        "interval around it, and that is what belongs on the slide.",
    ]
    return "\n".join(lines)


def render_absence(stats: AbsenceStats, counterfactual: AbsenceCounterfactual) -> str:
    """The markdown for what a zero in this pipeline actually means."""
    zero_share = stats.zeros / stats.total if stats.total else 0.0
    silent_share = stats.zeros_without_evidence / stats.zeros if stats.zeros else 0.0
    return "\n".join(
        [
            "## What a criterion score is, underneath",
            "",
            f"{stats.total} criterion scores were written across these rows. "
            f"**{100 * stats.share_at_three_levels:.1f}%** of them "
            f"({int(round(stats.share_at_three_levels * stats.total))} scores) are "
            "exactly 0.0, 0.5 or 1.0: the model is not using the 0..1 range it "
            "was given, it is answering no, maybe or yes.",
            "",
            f"**{stats.zeros}** of them ({100 * zero_share:.1f}%) are 0.0, and "
            f"**{stats.zeros_without_evidence}** of those ({100 * silent_share:.1f}%) "
            "reach the aggregator with no "
            "evidence attached at all -- no model-written quote and no hit from "
            "`expand_skill` either. Counted over the model's own answers in the "
            "cache rather than after the tools have added theirs, *every* 0.0 "
            "comes back without a quote and every non-zero score comes back with "
            "one. "
            "`SCORE_SYSTEM` asks for exactly this -- *\"If nothing in the resume "
            "supports the criterion, score 0.0 and return no quotes\"* -- so a CV "
            "that is silent on a criterion and a CV that plainly fails it reach "
            "`aggregate_scorecard` as the same number.",
            "",
            "That conflation is real, but it is not the reason the ordering is "
            "lost. Rebuilding the weighted sum over only the criteria that "
            "actually found evidence is the obvious repair, and it does not "
            "work:",
            "",
            "| Ordering | absence kept as 0.0 | absence dropped |",
            "|---|---:|---:|",
            f"| `Good Fit` over `Potential Fit` | "
            f"{counterfactual.kept_good_over_potential:.3f} | "
            f"{counterfactual.dropped_good_over_potential:.3f} |",
            f"| `Good Fit` over `No Fit` | {counterfactual.kept_good_over_no:.3f} | "
            f"{counterfactual.dropped_good_over_no:.3f} |",
            "",
            "The contested pair stays at chance, the separable one gets worse, "
            f"and {counterfactual.dropped_rows_lost} rows lose every criterion "
            "they had and drop out of the comparison entirely. The "
            "absent-as-zero signal is carrying most of what separates a fit of "
            "some kind from no fit, so removing it costs more than it returns.",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
