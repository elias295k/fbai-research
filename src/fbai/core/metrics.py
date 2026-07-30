"""Order-safe probability metrics with an explicit H/D/A class contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt

CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")


class ProbabilitySanity(TypedDict):
    """Diagnostics for a candidate H/D/A probability matrix."""

    correct_shape: bool
    finite: bool
    unit_interval: bool
    rows_sum_to_one: bool


class FoldMetrics(TypedDict):
    """Supported probabilistic metrics for one evaluation fold."""

    n_samples: int
    log_loss: float
    brier_score: float
    ece: float


class WeightedFoldSummary(TypedDict):
    """Sample-count-weighted summary across evaluation folds."""

    n_samples: int
    log_loss: float
    brier_score: float
    ece: float


def probability_sanity(probabilities: npt.ArrayLike) -> ProbabilitySanity:
    """Return strict diagnostics without clipping or renormalizing predictions."""

    values = np.asarray(probabilities, dtype=float)
    correct_shape = values.ndim == 2 and values.shape[1] == len(CLASS_ORDER)
    finite = bool(np.isfinite(values).all())
    unit_interval = finite and bool(((values >= 0.0) & (values <= 1.0)).all())
    rows_sum_to_one = (
        correct_shape
        and finite
        and bool(np.isclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8).all())
    )
    return {
        "correct_shape": correct_shape,
        "finite": finite,
        "unit_interval": unit_interval,
        "rows_sum_to_one": rows_sum_to_one,
    }


def validate_probabilities(probabilities: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Validate and return an H/D/A probability matrix unchanged."""

    values = np.asarray(probabilities, dtype=float)
    sanity = probability_sanity(values)
    failed: list[str] = []
    if not sanity["correct_shape"]:
        failed.append("correct_shape")
    if not sanity["finite"]:
        failed.append("finite")
    if not sanity["unit_interval"]:
        failed.append("unit_interval")
    if not sanity["rows_sum_to_one"]:
        failed.append("rows_sum_to_one")
    if failed:
        raise ValueError(f"Invalid probability matrix: {', '.join(failed)}")
    if values.shape[0] == 0:
        raise ValueError("Probability matrix must contain at least one prediction")
    return values.astype(np.float64, copy=False)


def one_hot(
    labels: Sequence[str],
    *,
    class_order: Sequence[str] = CLASS_ORDER,
) -> npt.NDArray[np.float64]:
    """Encode labels using an explicit class order."""

    if tuple(class_order) != CLASS_ORDER:
        raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
    class_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    encoded = np.zeros((len(labels), len(CLASS_ORDER)), dtype=np.float64)
    for row, label in enumerate(labels):
        if label not in class_index:
            raise ValueError(f"Unknown result label {label!r}; expected one of {CLASS_ORDER}")
        encoded[row, class_index[label]] = 1.0
    return encoded


def _validated_pair(
    labels: Sequence[str],
    probabilities: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    values = validate_probabilities(probabilities)
    targets = one_hot(labels)
    if len(targets) != len(values):
        raise ValueError(
            f"Label/prediction length mismatch: {len(targets)} labels and {len(values)} predictions"
        )
    return targets, values


def multiclass_log_loss(
    labels: Sequence[str],
    probabilities: npt.ArrayLike,
    *,
    epsilon: float = 1e-15,
) -> float:
    """Compute multiclass log loss in explicit H/D/A order."""

    targets, values = _validated_pair(labels, probabilities)
    true_probabilities = values[targets.astype(bool)]
    return float(-np.log(np.clip(true_probabilities, epsilon, 1.0)).mean())


def multiclass_brier_score(
    labels: Sequence[str],
    probabilities: npt.ArrayLike,
) -> float:
    """Compute the mean multiclass Brier score (sum across H/D/A)."""

    targets, values = _validated_pair(labels, probabilities)
    return float(np.square(values - targets).sum(axis=1).mean())


def expected_calibration_error(
    labels: Sequence[str],
    probabilities: npt.ArrayLike,
    *,
    n_bins: int = 10,
) -> float:
    """Compute top-label expected calibration error with equal-width bins."""

    if n_bins < 1:
        raise ValueError("n_bins must be at least one")
    targets, values = _validated_pair(labels, probabilities)
    predicted_class = values.argmax(axis=1)
    confidence = values.max(axis=1)
    correct = targets[np.arange(len(targets)), predicted_class]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.digitize(confidence, bin_edges[1:], right=True), n_bins - 1)

    ece = 0.0
    for current_bin in range(n_bins):
        mask = bin_index == current_bin
        if not mask.any():
            continue
        weight = float(mask.mean())
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += weight * abs(accuracy - mean_confidence)
    return ece


def evaluate_predictions(
    labels: Sequence[str],
    probabilities: npt.ArrayLike,
    *,
    n_bins: int = 10,
) -> FoldMetrics:
    """Evaluate valid H/D/A probability predictions."""

    values = validate_probabilities(probabilities)
    if len(labels) != len(values):
        raise ValueError("Label/prediction length mismatch")
    return {
        "n_samples": len(labels),
        "log_loss": multiclass_log_loss(labels, values),
        "brier_score": multiclass_brier_score(labels, values),
        "ece": expected_calibration_error(labels, values, n_bins=n_bins),
    }


def weighted_fold_summary(
    fold_metrics: Sequence[Mapping[str, float | int]],
) -> WeightedFoldSummary:
    """Aggregate fold metrics by sample count rather than by fold count."""

    if not fold_metrics:
        raise ValueError("At least one fold is required")
    total = sum(int(fold["n_samples"]) for fold in fold_metrics)
    if total <= 0:
        raise ValueError("Total fold sample count must be positive")

    weighted = {
        metric: sum(float(fold[metric]) * int(fold["n_samples"]) for fold in fold_metrics) / total
        for metric in ("log_loss", "brier_score", "ece")
    }
    return {
        "n_samples": total,
        "log_loss": weighted["log_loss"],
        "brier_score": weighted["brier_score"],
        "ece": weighted["ece"],
    }
