import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as LogisticRegressionStub

from eval.scoring_probe import (
    CriterionRow,
    absence_zero_share,
    auc,
    bootstrap_auc_ci,
    grouped_ceiling_auc,
    render_scoring_probe,
    scoring_probe,
)


def test_a_perfectly_ordered_pair_of_classes_scores_one():
    assert auc([0.9, 0.8], [0.2, 0.1]) == 1.0


def test_the_reverse_ordering_scores_zero_rather_than_being_folded_to_one():
    """0.4 has to stay readable as "worse than chance", not "as good as 0.6".

    The shipped pipeline sits at 0.482, and the whole point of reporting it is
    that it is below the coin flip, not merely far from 1.0.
    """
    assert auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_ties_count_as_half_a_win():
    assert auc([0.5], [0.5]) == 0.5
    # Four pairs: one tie at 0.5, three clean wins. (0.5 + 3) / 4.
    assert auc([0.5, 0.9], [0.5, 0.1]) == 0.875


def test_an_empty_class_has_no_auc_rather_than_a_misleading_half():
    assert np.isnan(auc([], [0.5]))


def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(60), np.zeros(60)]
    scores = np.r_[rng.normal(0.6, 1, 60), rng.normal(0.0, 1, 60)]

    point = auc(scores[labels == 1], scores[labels == 0])
    low, high = bootstrap_auc_ci(labels, scores, reps=500, seed=0)

    assert low < point < high


def test_a_smaller_sample_gives_a_wider_interval():
    """Why the Good-vs-Potential question cannot be settled on dev_300.

    149 rows put chance inside the interval no matter what the point estimate
    is, which is the same underpowered trap Day 4 hit with its sign test.
    """
    rng = np.random.default_rng(1)
    big_labels = np.r_[np.ones(300), np.zeros(300)]
    big_scores = np.r_[rng.normal(0.4, 1, 300), rng.normal(0.0, 1, 300)]
    small_labels, small_scores = big_labels[::6], big_scores[::6]

    big_low, big_high = bootstrap_auc_ci(big_labels, big_scores, reps=400, seed=0)
    small_low, small_high = bootstrap_auc_ci(small_labels, small_scores, reps=400, seed=0)

    assert (small_high - small_low) > (big_high - big_low)


def test_grouping_by_job_description_removes_the_leak_a_plain_split_rewards():
    """The measurement mistake this function exists to prevent.

    A feature that is nothing but the job description's identity looks
    predictive under a plain split, because the same JD lands in train and
    test with the same label. Grouped, it is worth nothing. Measured on the
    real split this gap was 0.701 against 0.577.
    """
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(30), 6)
    y = np.repeat(rng.integers(0, 2, 30), 6)
    # One column, perfectly collinear with the group, carrying no transferable signal.
    features = np.c_[groups.astype(float), rng.normal(size=len(y))]

    grouped = grouped_ceiling_auc(features, y, groups, folds=5, seed=0)
    ungrouped = grouped_ceiling_auc(features, y, np.arange(len(y)), folds=5, seed=0)

    assert grouped < ungrouped
    assert grouped < 0.75


def test_a_zero_with_no_evidence_is_counted_as_absence_not_as_failure():
    rows = [
        CriterionRow(score=0.0, has_evidence=False),
        CriterionRow(score=0.0, has_evidence=False),
        CriterionRow(score=0.5, has_evidence=True),
        CriterionRow(score=1.0, has_evidence=True),
    ]

    stats = absence_zero_share(rows)

    assert stats.total == 4
    assert stats.zeros == 2
    assert stats.zeros_without_evidence == 2
    assert stats.share_at_three_levels == 1.0


def test_the_three_level_share_only_counts_the_three_levels():
    rows = [CriterionRow(score=s, has_evidence=s > 0) for s in (0.0, 0.5, 1.0, 0.37)]

    assert absence_zero_share(rows).share_at_three_levels == 0.75


def test_the_probe_reports_the_shipped_number_beside_both_ceilings():
    rng = np.random.default_rng(3)
    n = 60
    labels = ["Good Fit"] * n + ["Potential Fit"] * n
    shipped = list(rng.normal(0.5, 0.2, 2 * n))
    groups = [f"jd{i % 25}" for i in range(2 * n)]
    features = rng.normal(size=(2 * n, 3))
    text_features = rng.normal(size=(2 * n, 3))

    result = scoring_probe(
        labels=labels,
        shipped_scores=shipped,
        groups=groups,
        criterion_features=features,
        text_features=text_features,
        reps=200,
        seed=0,
    )

    assert result.rows == 2 * n
    assert 0.0 <= result.shipped_auc <= 1.0
    assert result.ci_low < result.shipped_auc < result.ci_high
    assert "Good Fit" in render_scoring_probe(result)
    assert f"{result.shipped_auc:.3f}" in render_scoring_probe(result)


def test_the_probe_refuses_a_split_with_only_one_class():
    with pytest.raises(ValueError):
        scoring_probe(
            labels=["Good Fit"] * 10,
            shipped_scores=[0.5] * 10,
            groups=[f"jd{i}" for i in range(10)],
            criterion_features=np.zeros((10, 2)),
            text_features=np.zeros((10, 2)),
        )


def test_a_text_pipeline_is_fitted_inside_each_fold_not_once_over_everything():
    """The vectoriser must not see the held-out rows either.

    Fitting TF-IDF on all rows before splitting leaks the test fold's vocabulary,
    which inflates the very ceiling this number is supposed to bound.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(20), 5)
    y = np.repeat(rng.integers(0, 2, 20), 5)
    texts = [f"token{g} common words here" for g in groups]

    pipeline = make_pipeline(TfidfVectorizer(min_df=1), LogisticRegressionStub())
    score = grouped_ceiling_auc(texts, y, groups, estimator=pipeline, folds=5, seed=0)

    assert 0.0 <= score <= 1.0


def test_the_threshold_search_finds_the_best_pair_of_cut_points():
    """What recalibration alone is worth, with nothing else changed.

    The old measurement put this at +0.023 macro-F1 and concluded the missing
    accuracy is not in the cut points. The search has to stay in the repo for
    that claim to be re-checkable against whatever records exist later.
    """
    from eval.scoring_probe import best_thresholds

    # Cleanly separated by score: 0.9 good, 0.5 potential, 0.1 no fit.
    labels = ["Good Fit"] * 10 + ["Potential Fit"] * 10 + ["No Fit"] * 10
    scores = [0.9] * 10 + [0.5] * 10 + [0.1] * 10

    best = best_thresholds(labels, scores, step=0.01)

    assert best.macro_f1 == pytest.approx(1.0)
    assert best.potential_threshold < best.good_threshold


def test_the_threshold_search_reports_what_the_shipped_cuts_score_on_the_same_rows():
    from eval.scoring_probe import best_thresholds

    labels = ["Good Fit"] * 10 + ["No Fit"] * 10
    scores = [0.9] * 10 + [0.1] * 10

    best = best_thresholds(labels, scores, step=0.05)

    assert best.shipped_macro_f1 <= best.macro_f1
    assert best.gain == pytest.approx(best.macro_f1 - best.shipped_macro_f1)


def test_the_absence_counterfactual_is_computed_not_asserted():
    """Day 4 published a fairness figure that matched no completed run.

    The same trap applies here: "dropping the absent-as-zero scores moves the
    AUC from X to Y" is a number, and it has to come out of the records rather
    than out of a sentence somebody typed.
    """
    from eval.scoring_probe import absence_counterfactual

    dumped = [
        {
            "true_label": "Good Fit",
            "criteria": [
                {"score": 0.0, "has_evidence": False, "weight": 0.5},
                {"score": 1.0, "has_evidence": True, "weight": 0.5},
            ],
        },
        {
            "true_label": "Potential Fit",
            "criteria": [
                {"score": 0.0, "has_evidence": False, "weight": 0.5},
                {"score": 0.0, "has_evidence": True, "weight": 0.5},
            ],
        },
        {
            "true_label": "No Fit",
            "criteria": [{"score": 0.0, "has_evidence": False, "weight": 1.0}],
        },
    ]

    result = absence_counterfactual(dumped)

    # Kept: Good scores 0.5, Potential 0.0 -> Good wins.
    assert result.kept_good_over_potential == 1.0
    # Dropped: Good keeps only its 1.0, Potential only its 0.0 -> Good still wins.
    assert result.dropped_good_over_potential == 1.0
    # No Fit has nothing left once absence is dropped, so it leaves the comparison.
    assert result.dropped_rows_lost > 0


def test_a_row_with_nothing_left_after_dropping_absence_is_counted_not_scored():
    from eval.scoring_probe import absence_counterfactual

    dumped = [
        {"true_label": "Good Fit",
         "criteria": [{"score": 0.0, "has_evidence": False, "weight": 1.0}]},
        {"true_label": "Potential Fit",
         "criteria": [{"score": 0.5, "has_evidence": True, "weight": 1.0}]},
        {"true_label": "No Fit",
         "criteria": [{"score": 0.2, "has_evidence": True, "weight": 1.0}]},
    ]

    result = absence_counterfactual(dumped)

    assert result.dropped_rows_lost == 1


def test_power_rises_with_sample_size_and_falls_as_the_effect_shrinks():
    """Whether the one `test_500` run can settle the ordering at all.

    Day 4 published a fairness conclusion from a sign test with power 0.40. The
    same question has to be asked before spending the single run this project
    is allowed, and the answer has to be in the repo rather than in a message.
    """
    from eval.scoring_probe import auc_power

    assert auc_power(0.58, 75) < auc_power(0.58, 250)
    assert auc_power(0.55, 250) < auc_power(0.70, 250)
    assert 0.0 <= auc_power(0.50, 250) <= 0.10


def test_the_power_of_dev_300_on_this_effect_is_the_coin_flip_it_was_measured_to_be():
    from eval.scoring_probe import auc_power

    # 149 contested rows, roughly 75 per class, against the 0.577 text ceiling.
    assert auc_power(0.577, 75) < 0.5


def test_smallest_detectable_sample_reports_the_n_that_would_settle_it():
    from eval.scoring_probe import rows_needed

    needed = rows_needed(0.58, power=0.80)

    assert needed > 150
    assert rows_needed(0.70, power=0.80) < needed
