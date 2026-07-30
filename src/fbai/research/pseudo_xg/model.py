"""Fold-local Poisson pseudo-xG estimator and LR52-plus-pseudo-xG candidate."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbai.core.leakage import assert_model_inputs_safe
from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.data.schema import validate_canonical_frame
from fbai.features.schema import FEATURE_COLUMNS, FEATURE_TABLE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, TARGET_COLUMN
from fbai.research.pseudo_xg.config import PseudoXGConfig

PSEUDO_XG_PREDICTOR_COLUMNS: tuple[str, str] = ("sot", "off")


class PseudoXGModelError(ValueError):
    """Raised when pseudo-xG fitting or candidate input violates its contract."""


class PseudoXGConvergenceError(RuntimeError):
    """Raised when either experimental optimizer does not converge."""


@dataclass(frozen=True, slots=True)
class FittedPseudoXGEstimator:
    """A fitted fold-local estimator plus aggregate training metadata."""

    estimator: PoissonRegressor
    config: PseudoXGConfig
    training_match_rows: int
    training_side_rows: int
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class FittedPseudoXGCandidate:
    """The experimental LR64 pipeline and its explicit input contract."""

    pipeline: Pipeline
    config: PseudoXGConfig
    feature_columns: tuple[str, ...]
    sklearn_classes: tuple[str, ...]
    iterations: tuple[int, ...]
    converged: bool


def _side_training_rows(canonical: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "goals": canonical["FTHome"],
            "sot": canonical["HomeTarget"],
            "off": canonical["HomeShots"] - canonical["HomeTarget"],
        }
    )
    away = pd.DataFrame(
        {
            "goals": canonical["FTAway"],
            "sot": canonical["AwayTarget"],
            "off": canonical["AwayShots"] - canonical["AwayTarget"],
        }
    )
    rows = pd.concat([home, away], ignore_index=True).dropna()
    return rows.loc[rows["sot"].ge(0) & rows["off"].ge(0)].reset_index(drop=True)


def fit_pseudo_xg_estimator(
    canonical_train: pd.DataFrame,
    *,
    config: PseudoXGConfig | None = None,
) -> FittedPseudoXGEstimator:
    """Fit the source Poisson estimator using only the supplied training matches."""

    validate_canonical_frame(canonical_train)
    resolved = config or PseudoXGConfig()
    rows = _side_training_rows(canonical_train)
    if rows.empty:
        raise PseudoXGModelError("pseudo-xG estimator has no valid team-side training rows")
    estimator = PoissonRegressor(
        alpha=resolved.poisson_alpha,
        fit_intercept=resolved.poisson_fit_intercept,
        solver=resolved.poisson_solver,
        max_iter=resolved.poisson_max_iter,
        tol=resolved.poisson_tolerance,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(rows.loc[:, list(PSEUDO_XG_PREDICTOR_COLUMNS)], rows["goals"])
    convergence_warnings = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    iterations = int(estimator.n_iter_)
    converged = not convergence_warnings and iterations < resolved.poisson_max_iter
    if not converged:
        raise PseudoXGConvergenceError(
            f"pseudo-xG Poisson estimator failed within {resolved.poisson_max_iter} iterations"
        )
    return FittedPseudoXGEstimator(
        estimator=estimator,
        config=resolved,
        training_match_rows=len(canonical_train),
        training_side_rows=len(rows),
        iterations=iterations,
        converged=True,
    )


def predict_match_pseudo_xg(
    fitted: FittedPseudoXGEstimator,
    *,
    shots_on_target: pd.Series,
    shots: pd.Series,
) -> np.ndarray:
    """Estimate completed-match pseudo-xG, preserving invalid rows as missing."""

    predictors = pd.DataFrame(
        {
            "sot": pd.to_numeric(shots_on_target, errors="coerce"),
            "off": pd.to_numeric(shots, errors="coerce")
            - pd.to_numeric(shots_on_target, errors="coerce"),
        }
    )
    valid = predictors.notna().all(axis=1) & predictors["sot"].ge(0) & predictors["off"].ge(0)
    output = np.full(len(predictors), np.nan, dtype=np.float64)
    if bool(valid.any()):
        output[valid.to_numpy()] = fitted.estimator.predict(predictors.loc[valid])
    return output


def candidate_feature_columns() -> tuple[str, ...]:
    """Return the explicit 64-feature experimental input tuple."""

    from fbai.research.pseudo_xg.features import PSEUDO_XG_FEATURE_COLUMNS

    return (*FEATURE_COLUMNS, *PSEUDO_XG_FEATURE_COLUMNS)


def _select_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = candidate_feature_columns()
    expected = set(FEATURE_TABLE_COLUMNS).union(feature_columns)
    missing = [column for column in feature_columns if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing:
        raise PseudoXGModelError(f"Missing pseudo-xG candidate features: {', '.join(missing)}")
    if extra:
        raise PseudoXGModelError(f"Unexpected pseudo-xG candidate columns: {', '.join(extra)}")
    assert_model_inputs_safe(feature_columns, approved_pre_features=feature_columns)
    selected = frame.loc[:, list(feature_columns)].copy(deep=True)
    for column in feature_columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    if np.isinf(selected.to_numpy(dtype=np.float64, na_value=np.nan)).any():
        raise PseudoXGModelError("pseudo-xG candidate inputs contain infinite values")
    return selected


def fit_pseudo_xg_candidate(
    train_frame: pd.DataFrame,
    *,
    config: PseudoXGConfig | None = None,
) -> FittedPseudoXGCandidate:
    """Fit source-equivalent LR on the exact 52 plus 12 experimental features."""

    resolved = config or PseudoXGConfig()
    selected = _select_candidate_features(train_frame)
    if TARGET_COLUMN not in train_frame.columns or train_frame[TARGET_COLUMN].isna().any():
        raise PseudoXGModelError(f"Training frame requires complete {TARGET_COLUMN}")
    targets = train_frame[TARGET_COLUMN].astype(str)
    invalid = sorted(set(targets).difference(CLASS_ORDER))
    if invalid or frozenset(targets) != frozenset(CLASS_ORDER):
        raise PseudoXGModelError("pseudo-xG candidate training requires all H/D/A classes")
    classifier = LogisticRegression(
        C=resolved.logistic_c,
        solver=resolved.logistic_solver,
        max_iter=resolved.logistic_max_iter,
        fit_intercept=resolved.logistic_fit_intercept,
        class_weight=resolved.logistic_class_weight,
        random_state=resolved.logistic_random_state,
        tol=resolved.logistic_tolerance,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=resolved.imputation)),
            ("scaler", StandardScaler()),
            ("logistic", classifier),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(selected, targets)
    convergence_warnings = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    iterations = tuple(int(value) for value in classifier.n_iter_)
    converged = not convergence_warnings and all(
        iteration < resolved.logistic_max_iter for iteration in iterations
    )
    if not converged:
        raise PseudoXGConvergenceError(
            f"pseudo-xG LR candidate failed within {resolved.logistic_max_iter} iterations"
        )
    return FittedPseudoXGCandidate(
        pipeline=pipeline,
        config=resolved,
        feature_columns=candidate_feature_columns(),
        sklearn_classes=tuple(str(value) for value in classifier.classes_),
        iterations=iterations,
        converged=True,
    )


def predict_pseudo_xg_candidate_proba(
    fitted: FittedPseudoXGCandidate,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict candidate probabilities in the explicit public H/D/A order."""

    if fitted.feature_columns != candidate_feature_columns():
        raise PseudoXGModelError("fitted pseudo-xG candidate feature tuple is invalid")
    selected = _select_candidate_features(frame)
    raw = np.asarray(fitted.pipeline.predict_proba(selected), dtype=np.float64)
    classifier = cast(LogisticRegression, fitted.pipeline.named_steps["logistic"])
    classes = tuple(str(value) for value in classifier.classes_)
    order = [classes.index(label) for label in CLASS_ORDER]
    probabilities = validate_probabilities(raw[:, order])
    return pd.DataFrame(probabilities, index=frame.index.copy(), columns=PROBABILITY_COLUMNS)
