from __future__ import annotations

import math

import numpy as np
import pytest

from fbai.core.metrics import (
    CLASS_ORDER,
    evaluate_predictions,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    one_hot,
    probability_sanity,
    validate_probabilities,
    weighted_fold_summary,
)


def test_class_order_is_explicit_home_draw_away() -> None:
    assert CLASS_ORDER == ("H", "D", "A")
    np.testing.assert_array_equal(
        one_hot(["H", "D", "A"]),
        np.eye(3),
    )


def test_perfect_predictions_have_near_zero_loss() -> None:
    labels = ["H", "D", "A"]
    probabilities = np.eye(3)

    assert multiclass_log_loss(labels, probabilities) == pytest.approx(0.0)
    assert multiclass_brier_score(labels, probabilities) == pytest.approx(0.0)
    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.0)


def test_uniform_predictions_have_expected_log_loss() -> None:
    labels = ["H", "D", "A"]
    probabilities = np.full((3, 3), 1.0 / 3.0)

    assert multiclass_log_loss(labels, probabilities) == pytest.approx(math.log(3.0))


def test_hda_order_is_not_inferred_from_observed_labels() -> None:
    labels = ["A", "H"]
    probabilities = np.array([[0.05, 0.05, 0.90], [0.80, 0.10, 0.10]])

    metrics = evaluate_predictions(labels, probabilities)

    assert metrics["log_loss"] == pytest.approx(-0.5 * (math.log(0.90) + math.log(0.80)))


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[0.5, 0.5]]),
        np.array([[0.8, 0.3, -0.1]]),
        np.array([[0.5, 0.4, 0.2]]),
        np.array([[np.nan, 0.5, 0.5]]),
        np.array([[np.inf, 0.0, 0.0]]),
    ],
)
def test_invalid_probabilities_raise_instead_of_being_repaired(
    probabilities: np.ndarray,
) -> None:
    original = probabilities.copy()

    with pytest.raises(ValueError, match="Invalid probability matrix"):
        validate_probabilities(probabilities)

    np.testing.assert_equal(probabilities, original)


def test_probability_sanity_reports_individual_failures() -> None:
    sanity = probability_sanity([[0.6, 0.3, 0.3]])

    assert sanity == {
        "correct_shape": True,
        "finite": True,
        "unit_interval": True,
        "rows_sum_to_one": False,
    }


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown result label"):
        one_hot(["W"])


def test_label_prediction_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_predictions(["H"], np.full((2, 3), 1.0 / 3.0))


def test_weighted_fold_summary_uses_sample_counts() -> None:
    summary = weighted_fold_summary(
        [
            {"n_samples": 10, "log_loss": 1.0, "brier_score": 0.6, "ece": 0.2},
            {"n_samples": 30, "log_loss": 0.5, "brier_score": 0.3, "ece": 0.1},
        ]
    )

    assert summary["n_samples"] == 40
    assert summary["log_loss"] == pytest.approx(0.625)
    assert summary["brier_score"] == pytest.approx(0.375)
    assert summary["ece"] == pytest.approx(0.125)
