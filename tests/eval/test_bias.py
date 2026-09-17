import pytest

from eval.bias import (
    IDENTITIES,
    BiasRow,
    Identity,
    inject_identity,
    render_bias,
    sign_test_p,
    swap_school,
)

CV = "Professional Summary Six years of Python. Education BS from Ohio State University."


def test_the_identities_differ_on_both_axes_spec_section_7_names():
    assert len(IDENTITIES) >= 2
    assert len({identity.name for identity in IDENTITIES}) == len(IDENTITIES)
    assert len({identity.pronoun for identity in IDENTITIES}) >= 2


def test_an_identity_is_prepended_without_disturbing_the_original_text():
    injected = inject_identity(CV, Identity(name="James Miller", pronoun="he/him"))

    assert injected.endswith(CV)
    assert "James Miller" in injected
    assert "he/him" in injected


def test_injection_is_the_only_difference_between_two_arms_of_the_same_cv():
    first = inject_identity(CV, IDENTITIES[0])
    second = inject_identity(CV, IDENTITIES[1])

    assert first != second
    assert first[len(first) - len(CV):] == second[len(second) - len(CV):]


def test_a_school_is_replaced_and_the_replacement_is_reported():
    swapped, changed = swap_school(CV, "Kabul Polytechnic University")

    assert changed is True
    assert "Ohio State University" not in swapped
    assert "Kabul Polytechnic University" in swapped
    assert "Six years of Python" in swapped


def test_swapping_a_school_does_not_swallow_the_words_around_it():
    """The obvious looser regex eats the sentence and the arm measures deletion.

    Matching "any four words then University" starts at "Education" here, so the
    swap would silently drop "Education BS from" as well. Every word before the
    institution keyword has to be capitalised for this reason.
    """
    swapped, _ = swap_school(CV, "MIT")

    assert "Education BS from" in swapped
    assert swapped == "Professional Summary Six years of Python. Education BS from MIT."


def test_a_cv_naming_no_school_is_left_alone_and_says_so():
    swapped, changed = swap_school("Professional Summary Six years of Python.", "MIT")

    assert changed is False
    assert swapped == "Professional Summary Six years of Python."


def test_the_report_leads_with_how_many_scores_moved_at_all():
    rows = [
        BiasRow(row_index=0, arm="identity", variant_a="James Miller",
                variant_b="Aisha Okonkwo", score_a=0.62, score_b=0.62,
                label_a="Potential Fit", label_b="Potential Fit"),
        BiasRow(row_index=1, arm="identity", variant_a="James Miller",
                variant_b="Aisha Okonkwo", score_a=0.71, score_b=0.64,
                label_a="Good Fit", label_b="Potential Fit"),
    ]

    text = render_bias(rows, "identity")

    assert "1" in text          # one label flipped
    assert "0.07" in text       # the largest score move
    assert "injected" in text.lower()


def test_an_arm_where_nothing_moved_says_so_rather_than_printing_an_empty_table():
    rows = [
        BiasRow(row_index=0, arm="school", variant_a="Ohio State University",
                variant_b="Kabul Polytechnic University", score_a=0.5, score_b=0.5,
                label_a="Potential Fit", label_b="Potential Fit"),
    ]

    text = render_bias(rows, "school")

    assert "0 of 1" in text or "no score changed" in text.lower()


def test_a_bias_row_computes_its_own_deltas():
    row = BiasRow(row_index=0, arm="identity", variant_a="A", variant_b="B",
                  score_a=0.7, score_b=0.55, label_a="Good Fit", label_b="Potential Fit")

    assert row.delta == pytest.approx(-0.15)
    assert row.flipped is True


def test_the_report_separates_a_biased_pipeline_from_an_unstable_one():
    """Direction and instability are independent defects; the report must not conflate them.

    Measured over all 261 eligible pairs on 2026-09-17, cold and warm runs identical:

    - school:   72 of 261 moved, 46 down / 26 up, sign test p = 0.024, 24 labels
      flipped. The moves share a direction -- swapping MIT for Kabul Polytechnic
      lowers the score more often than chance allows.
    - identity: 71 of 261 moved, 33 down / 38 up, sign test p = 0.635, 21 labels
      flipped. Instability of the same size, with no shared direction.

    An earlier version of this docstring cited "15 of 50, 7 up and 8 down, mean
    delta +0.006 -- not bias". Those figures matched no completed run; the n=50
    school run that did complete gave 11 down / 4 up. At n=50 the sign test had
    power 0.40, so its p = 0.119 was underpowered, not evidence of fairness.
    """
    rows = [
        BiasRow(row_index=0, arm="school", variant_a="A", variant_b="B",
                score_a=0.30, score_b=0.50, label_a="No Fit", label_b="Potential Fit"),
        BiasRow(row_index=1, arm="school", variant_a="A", variant_b="B",
                score_a=0.50, score_b=0.30, label_a="Potential Fit", label_b="No Fit"),
    ]

    text = render_bias(rows, "school")

    assert "1 up" in text and "1 down" in text
    assert "+0.000" in text          # mean delta: the two cancel
    assert "0.200" in text           # mean absolute move: they do not


def test_a_lopsided_arm_is_reported_as_directional():
    """A small mean delta must not be readable as fairness when the moves agree.

    These nine rows each move +0.10, so mean delta is small in absolute terms but
    every move points the same way. The report must say so rather than let the
    magnitude speak for the direction.
    """
    rows = [
        BiasRow(row_index=i, arm="school", variant_a="A", variant_b="B",
                score_a=0.40, score_b=0.50, label_a="Potential Fit", label_b="Potential Fit")
        for i in range(9)
    ]

    text = render_bias(rows, "school")

    assert "9 up, 0 down" in text
    assert "the moves share a direction" in text
    assert "p = **0.004**" in text


def test_a_balanced_arm_is_not_reported_as_directional():
    """The mirror case: equal moves both ways must not be called bias."""
    rows = [
        BiasRow(row_index=i, arm="identity", variant_a="A", variant_b="B",
                score_a=0.40, score_b=0.50 if i % 2 else 0.30,
                label_a="Potential Fit", label_b="Potential Fit")
        for i in range(10)
    ]

    text = render_bias(rows, "identity")

    assert "no shared direction at this sample size" in text


def test_the_sign_test_drops_ties_rather_than_counting_them_as_agreement():
    """Ties carry no direction. Counting them would dilute a real effect to nothing."""
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(72, 46) == pytest.approx(0.024, abs=0.001)
    assert sign_test_p(71, 33) == pytest.approx(0.635, abs=0.001)
    assert sign_test_p(10, 5) == pytest.approx(1.0)
