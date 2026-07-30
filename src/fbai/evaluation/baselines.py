"""Non-market uniform and train-fitted class-prior probability baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.models.logistic import PROBABILITY_COLUMNS, TARGET_COLUMN


class BaselineTargetError(ValueError):
    """Raised when a training-prior baseline cannot fit H/D/A frequencies."""


def _probability_frame(
    probabilities: np.ndarray,
    *,
    index: pd.Index,
) -> pd.DataFrame:
    values = validate_probabilities(probabilities)
    return pd.DataFrame(values, index=index.copy(), columns=PROBABILITY_COLUMNS)


def uniform_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exact one-third H/D/A probabilities for every input row."""

    probabilities = np.full((len(frame), len(CLASS_ORDER)), 1.0 / 3.0, dtype="float64")
    return _probability_frame(probabilities, index=frame.index)


@dataclass(frozen=True, slots=True)
class TrainingPriorBaseline:
    """Three-class probabilities fitted from one training fold only."""

    probabilities: tuple[float, float, float]
    class_order: tuple[str, str, str] = CLASS_ORDER

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Repeat the fitted H/D/A prior without reading test labels."""

        values = np.tile(np.asarray(self.probabilities, dtype="float64"), (len(frame), 1))
        return _probability_frame(values, index=frame.index)


def fit_training_prior(
    train_frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> TrainingPriorBaseline:
    """Fit H/D/A frequencies from the current training fold."""

    if target_column not in train_frame.columns:
        raise BaselineTargetError(f"Training frame is missing {target_column}")
    target = train_frame[target_column]
    if target.isna().any():
        raise BaselineTargetError(f"{target_column} contains missing values")
    invalid = sorted(set(target.astype(str)).difference(CLASS_ORDER))
    if invalid:
        raise BaselineTargetError(f"{target_column} contains values outside H/D/A: {invalid}")
    counts = target.astype(str).value_counts()
    missing = [label for label in CLASS_ORDER if int(counts.get(label, 0)) == 0]
    if missing:
        raise BaselineTargetError(
            "Training-prior baseline requires all H/D/A classes; missing: " + ", ".join(missing)
        )
    total = float(len(target))
    probabilities = (
        float(counts["H"] / total),
        float(counts["D"] / total),
        float(counts["A"] / total),
    )
    return TrainingPriorBaseline(probabilities=probabilities)
