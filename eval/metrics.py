"""Macro-F1 and a confusion matrix over the three dataset labels.

Spec section 7 makes macro-F1 the headline number, so it is written out here in
full rather than imported: the figure on the slide should be twenty lines a
reviewer can check by hand. `tests/eval/test_metrics.py` cross-checks it against
`sklearn.metrics.f1_score`, which is how the hand-written version earns the right
to be the one that runs.

Everything here is a pure function over `(true_label, predicted_label)` pairs of
*label values* -- the strings the dataset uses -- because the records these come
from are JSON on disk.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from src.contracts.screening import FitLabel

LABELS: tuple[str, ...] = (
    FitLabel.GOOD_FIT.value,
    FitLabel.POTENTIAL_FIT.value,
    FitLabel.NO_FIT.value,
)

Pair = tuple[str, str]


class ClassScore(BaseModel):
    """One label's precision, recall and F1, with the counts behind them."""

    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


def _check(pairs: Sequence[Pair]) -> None:
    for true_label, predicted in pairs:
        for label in (true_label, predicted):
            if label not in LABELS:
                raise ValueError(f"not one of the three dataset labels: {label!r}")


def confusion_matrix(pairs: Sequence[Pair]) -> dict[str, dict[str, int]]:
    """Counts indexed truth-first: `matrix[true][predicted]`.

    Every cell is present even at zero. A confusion matrix with missing cells
    invites the reader to assume the missing ones were never possible.
    """
    _check(pairs)
    matrix = {truth: {prediction: 0 for prediction in LABELS} for truth in LABELS}
    for true_label, predicted in pairs:
        matrix[true_label][predicted] += 1
    return matrix


def per_class(pairs: Sequence[Pair]) -> dict[str, ClassScore]:
    """Precision, recall and F1 for each label.

    A 0/0 precision or recall is 0.0, not an error and not 1.0 -- the same
    convention as scikit-learn's `zero_division=0`, so the two agree.
    """
    matrix = confusion_matrix(pairs)
    scores: dict[str, ClassScore] = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in LABELS)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        total = precision + recall
        scores[label] = ClassScore(
            precision=precision,
            recall=recall,
            f1=(2 * precision * recall / total) if total else 0.0,
            support=support,
            predicted=predicted,
        )
    return scores


def macro_f1(pairs: Sequence[Pair]) -> float:
    """The unweighted mean of the three per-class F1 scores.

    Unweighted on purpose: the split is imbalanced (No Fit is half of it) and a
    weighted average would let a model that never predicts `Good Fit` look fine.
    """
    scores = per_class(pairs)
    return sum(score.f1 for score in scores.values()) / len(LABELS)
