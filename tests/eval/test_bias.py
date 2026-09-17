import pytest

from eval.bias import (
    IDENTITIES,
    BiasRow,
    Identity,
    inject_identity,
    render_bias,
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
    """Direction is the whole point of the arm.

    Scores that all move the same way are bias. Scores that move as much but in
    both directions are instability, which is a different defect with a different
    fix. Measured 2026-09-17: the school arm moved 15 of 50 scores, 7 up and 8
    down, mean delta +0.006 and mean absolute delta 0.130 -- not bias.
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
