"""Source-verified LR52 fitting and explicit H/D/A probability prediction."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbai.core.leakage import assert_model_inputs_safe
from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.preprocessing import LR52InputError, build_lr52_preprocessor, select_lr52_features

PROBABILITY_CLASS_ORDER: tuple[str, str, str] = CLASS_ORDER
PROBABILITY_COLUMNS: tuple[str, str, str] = (
    "probability_home",
    "probability_draw",
    "probability_away",
)
TARGET_COLUMN = "target_1x2"


class LR52TargetError(ValueError):
    """Raised when LR52 training targets violate the H/D/A contract."""


class LR52ConvergenceError(RuntimeError):
    """Raised when the source-verified optimizer does not converge."""


@dataclass(frozen=True, slots=True)
class LR52Config:
    """Immutable source-verified Logistic Regression configuration."""

    solver: str = "lbfgs"
    penalty: str = "l2"
    regularization_c: float = 1.0
    max_iter: int = 2000
    fit_intercept: bool = True
    class_weight: str | None = None
    random_state: int = 42
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.solver != "lbfgs":
            raise ValueError("LR52 solver is fixed to lbfgs")
        if self.penalty != "l2":
            raise ValueError("LR52 effective penalty is fixed to l2")
        if self.class_weight is not None:
            raise ValueError("LR52 class_weight is fixed to None")
        if self.regularization_c <= 0.0:
            raise ValueError("LR52 regularization_c must be positive")
        if self.max_iter < 1:
            raise ValueError("LR52 max_iter must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("LR52 tolerance must be positive")

    @property
    def identifier(self) -> str:
        """Return the stable public configuration identifier."""

        return "lr52_median_standardized_lbfgs_l2_c1_iter2000_seed42"

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        """Return JSON-safe configuration metadata."""

        return {key: value for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class FittedLR52:
    """Fitted pipeline plus auditable configuration and preprocessing state."""

    pipeline: Pipeline
    config: LR52Config
    feature_columns: tuple[str, ...]
    sklearn_classes: tuple[str, ...]
    iterations: tuple[int, ...]
    converged: bool

    @property
    def imputer_statistics(self) -> tuple[float, ...]:
        imputer = cast(SimpleImputer, self.pipeline.named_steps["imputer"])
        return tuple(float(value) for value in imputer.statistics_)

    @property
    def scaler_mean(self) -> tuple[float, ...]:
        scaler = cast(StandardScaler, self.pipeline.named_steps["scaler"])
        return tuple(float(value) for value in scaler.mean_)

    @property
    def scaler_scale(self) -> tuple[float, ...]:
        scaler = cast(StandardScaler, self.pipeline.named_steps["scaler"])
        return tuple(float(value) for value in scaler.scale_)


def build_lr52_pipeline(config: LR52Config | None = None) -> Pipeline:
    """Build an unfitted median → scaling → multinomial LR pipeline.

    The verified source relied on scikit-learn's L2 default. Newer
    scikit-learn versions deprecate the explicit ``penalty`` parameter while
    retaining the same L2 behavior when it is omitted, so the effective
    penalty is recorded in :class:`LR52Config` but not passed explicitly.
    """

    resolved = config or LR52Config()
    preprocessor = build_lr52_preprocessor()
    classifier = LogisticRegression(
        C=resolved.regularization_c,
        solver=resolved.solver,
        max_iter=resolved.max_iter,
        fit_intercept=resolved.fit_intercept,
        class_weight=resolved.class_weight,
        random_state=resolved.random_state,
        tol=resolved.tolerance,
    )
    return Pipeline(steps=[*preprocessor.steps, ("logistic", classifier)])


def _validated_targets(frame: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in frame.columns:
        raise LR52TargetError(f"Training frame is missing {TARGET_COLUMN}")
    target = frame[TARGET_COLUMN]
    if target.isna().any():
        raise LR52TargetError(f"{TARGET_COLUMN} contains missing values")
    invalid = sorted(set(target.astype(str)).difference(PROBABILITY_CLASS_ORDER))
    if invalid:
        raise LR52TargetError(f"{TARGET_COLUMN} contains values outside H/D/A: {invalid}")
    present = frozenset(target.astype(str))
    required = frozenset(PROBABILITY_CLASS_ORDER)
    if present != required:
        missing = ", ".join(sorted(required.difference(present)))
        raise LR52TargetError(f"LR52 training requires all H/D/A classes; missing: {missing}")
    return target.astype("string").copy(deep=True)


def fit_lr52(
    train_frame: pd.DataFrame,
    *,
    config: LR52Config | None = None,
) -> FittedLR52:
    """Fit LR52 with preprocessing learned exclusively from ``train_frame``."""

    selected = select_lr52_features(train_frame)
    target = _validated_targets(train_frame)
    resolved = config or LR52Config()
    pipeline = build_lr52_pipeline(resolved)

    # Keep this guard adjacent to fit: a suffix alone is not authorization.
    assert_model_inputs_safe(FEATURE_COLUMNS, approved_pre_features=FEATURE_COLUMNS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(selected, target)
    convergence_warnings = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    classifier = cast(LogisticRegression, pipeline.named_steps["logistic"])
    iterations = tuple(int(value) for value in classifier.n_iter_)
    converged = not convergence_warnings and all(
        iteration < resolved.max_iter for iteration in iterations
    )
    if not converged:
        raise LR52ConvergenceError(f"LR52 failed to converge within {resolved.max_iter} iterations")
    classes = tuple(str(value) for value in classifier.classes_)
    if frozenset(classes) != frozenset(PROBABILITY_CLASS_ORDER):
        raise LR52TargetError(f"Fitted class set is invalid: {classes}")
    return FittedLR52(
        pipeline=pipeline,
        config=resolved,
        feature_columns=FEATURE_COLUMNS,
        sklearn_classes=classes,
        iterations=iterations,
        converged=True,
    )


def predict_lr52_proba(model: FittedLR52, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict three named probabilities in explicit H, D, A order."""

    if model.feature_columns != FEATURE_COLUMNS:
        raise LR52InputError("Fitted model does not use the canonical LR52 feature tuple")
    selected = select_lr52_features(frame)
    raw = np.asarray(model.pipeline.predict_proba(selected), dtype="float64")
    try:
        order = [model.sklearn_classes.index(label) for label in PROBABILITY_CLASS_ORDER]
    except ValueError as exc:
        raise LR52TargetError("Fitted model does not contain all H/D/A classes") from exc
    probabilities = validate_probabilities(raw[:, order])
    return pd.DataFrame(probabilities, index=frame.index.copy(), columns=PROBABILITY_COLUMNS)
