import pytest

from eval.metrics import LABELS, confusion_matrix, macro_f1, per_class

GOOD, POTENTIAL, NO = "Good Fit", "Potential Fit", "No Fit"

# Hand-computed below; do not regenerate these from the implementation.
PAIRS = [
    (GOOD, GOOD),
    (GOOD, NO),
    (POTENTIAL, POTENTIAL),
    (NO, NO),
    (NO, POTENTIAL),
]


def test_the_three_labels_are_the_dataset_labels_in_a_fixed_order():
    assert LABELS == (GOOD, POTENTIAL, NO)


def test_confusion_matrix_is_indexed_truth_then_prediction():
    matrix = confusion_matrix(PAIRS)

    assert matrix[GOOD][GOOD] == 1
    assert matrix[GOOD][NO] == 1
    assert matrix[GOOD][POTENTIAL] == 0
    assert matrix[POTENTIAL][POTENTIAL] == 1
    assert matrix[NO][NO] == 1
    assert matrix[NO][POTENTIAL] == 1


def test_every_cell_exists_even_when_nothing_landed_in_it():
    matrix = confusion_matrix([(GOOD, GOOD)])

    assert set(matrix) == set(LABELS)
    assert all(set(row) == set(LABELS) for row in matrix.values())
    assert matrix[NO][POTENTIAL] == 0


def test_per_class_precision_recall_and_f1_are_hand_checkable():
    scores = per_class(PAIRS)

    # Good Fit: predicted once, correct once, one true Good Fit missed.
    assert scores[GOOD].precision == pytest.approx(1.0)
    assert scores[GOOD].recall == pytest.approx(0.5)
    assert scores[GOOD].f1 == pytest.approx(2 / 3)
    assert scores[GOOD].support == 2
    assert scores[GOOD].predicted == 1

    # Potential Fit: predicted twice, correct once, nothing missed.
    assert scores[POTENTIAL].precision == pytest.approx(0.5)
    assert scores[POTENTIAL].recall == pytest.approx(1.0)
    assert scores[POTENTIAL].f1 == pytest.approx(2 / 3)

    # No Fit: predicted twice, correct once, one missed.
    assert scores[NO].precision == pytest.approx(0.5)
    assert scores[NO].recall == pytest.approx(0.5)
    assert scores[NO].f1 == pytest.approx(0.5)


def test_macro_f1_is_the_unweighted_mean_of_the_three():
    assert macro_f1(PAIRS) == pytest.approx((2 / 3 + 2 / 3 + 0.5) / 3)


def test_a_class_nobody_predicted_scores_zero_rather_than_dividing_by_zero():
    scores = per_class([(GOOD, NO), (POTENTIAL, NO), (NO, NO)])

    assert scores[GOOD].precision == 0.0
    assert scores[GOOD].recall == 0.0
    assert scores[GOOD].f1 == 0.0
    assert scores[GOOD].predicted == 0


def test_an_empty_run_is_zero_everywhere_rather_than_an_exception():
    assert macro_f1([]) == 0.0
    assert per_class([])[GOOD].support == 0


def test_an_unknown_label_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="Maybe Fit"):
        confusion_matrix([(GOOD, "Maybe Fit")])


def test_it_agrees_with_scikit_learn_on_a_larger_mix():
    """The hand-written arithmetic has to match the standard implementation.

    Twenty auditable lines are only worth having if they are also correct.
    """
    from sklearn.metrics import f1_score

    truth = [GOOD, GOOD, GOOD, POTENTIAL, POTENTIAL, NO, NO, NO, NO, POTENTIAL]
    predictions = [GOOD, NO, POTENTIAL, POTENTIAL, NO, NO, NO, GOOD, POTENTIAL, POTENTIAL]
    pairs = list(zip(truth, predictions))

    expected = f1_score(
        truth, predictions, labels=list(LABELS), average="macro", zero_division=0
    )

    assert macro_f1(pairs) == pytest.approx(expected)
