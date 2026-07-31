"""Source-defined logistic classifiers for temporal graph features."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbai.core.leakage import assert_model_inputs_safe
from fbai.core.metrics import CLASS_ORDER, validate_probabilities
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS
from fbai.research.graph_model.config import (
    GRAPH_CLASSIFIER_CONFIG,
    GRAPH_ONLY_CONTEXT,
    LR52_GRAPH_CONTEXT,
    GraphClassifierConfig,
)


class GraphModelInputError(ValueError):
    """Raised when classifier inputs violate the graph-model contract."""


class GraphModelTrainingError(RuntimeError):
    """Raised when the source classifier cannot produce a valid fit."""


@dataclass(frozen=True, slots=True)
class FittedGraphClassifier:
    """Fitted aggregate-only graph classifier state."""

    pipeline: Pipeline
    config: GraphClassifierConfig
    context: str
    feature_columns: tuple[str, ...]
    graph_feature_columns: tuple[str, ...]
    sklearn_classes: tuple[str, ...]
    iterations: tuple[int, ...]
    converged: bool

    @property
    def parameter_count(self) -> int:
        return len(CLASS_ORDER) * (len(self.feature_columns) + 1)

    def state_fingerprint(self) -> str:
        """Return a deterministic digest without exposing learned values."""

        classifier = cast(LogisticRegression, self.pipeline.named_steps["logistic"])
        digest = hashlib.sha256()
        digest.update(classifier.coef_.tobytes())
        digest.update(classifier.intercept_.tobytes())
        return digest.hexdigest()


def graph_classifier_feature_columns(
    graph_columns: tuple[str, ...],
    *,
    context: str,
) -> tuple[str, ...]:
    """Return the exact source input order for one internal classifier context."""

    if not graph_columns or len(graph_columns) != len(set(graph_columns)):
        raise GraphModelInputError("graph feature columns must be non-empty and unique")
    if not all(column.startswith("graph_") and column.endswith("_pre") for column in graph_columns):
        raise GraphModelInputError("graph feature columns must use the graph_*_pre contract")
    if context == GRAPH_ONLY_CONTEXT:
        return graph_columns
    if context == LR52_GRAPH_CONTEXT:
        return (*FEATURE_COLUMNS, *graph_columns)
    raise GraphModelInputError(f"unsupported graph classifier context: {context}")


def _selected_numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise GraphModelInputError("graph classifier input must be a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise GraphModelInputError(f"graph classifier is missing: {', '.join(missing)}")
    for column in columns:
        if not is_numeric_dtype(frame[column].dtype):
            raise GraphModelInputError(f"graph classifier feature {column} must be numeric")
    selected = frame.loc[:, list(columns)].copy(deep=True)
    numeric = selected.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(numeric).any():
        raise GraphModelInputError("graph classifier features must not contain infinity")
    return selected


def _validated_targets(frame: pd.DataFrame) -> pd.Series:
    if "target_1x2" not in frame.columns:
        raise GraphModelInputError("graph classifier requires target_1x2")
    labels = frame["target_1x2"].astype(str)
    invalid = sorted(set(labels).difference(CLASS_ORDER))
    if invalid:
        raise GraphModelInputError(f"graph classifier labels are outside H/D/A: {invalid}")
    missing = frozenset(CLASS_ORDER).difference(labels)
    if missing:
        raise GraphModelInputError(
            f"graph classifier training is missing classes: {', '.join(sorted(missing))}"
        )
    return labels.copy(deep=True)


def _build_pipeline(config: GraphClassifierConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=config.regularization_c,
                    solver=config.solver,
                    max_iter=config.max_iter,
                    fit_intercept=config.fit_intercept,
                    class_weight=config.class_weight,
                    random_state=config.random_seed,
                    tol=config.tolerance,
                ),
            ),
        ]
    )


def fit_graph_classifier(
    train_frame: pd.DataFrame,
    graph_columns: tuple[str, ...],
    *,
    context: str,
    config: GraphClassifierConfig = GRAPH_CLASSIFIER_CONFIG,
) -> FittedGraphClassifier:
    """Fit fold-local preprocessing and source logistic regression."""

    feature_columns = graph_classifier_feature_columns(graph_columns, context=context)
    assert_model_inputs_safe(
        feature_columns,
        approved_pre_features=feature_columns,
    )
    selected = _selected_numeric(train_frame, feature_columns)
    labels = _validated_targets(train_frame)
    pipeline = _build_pipeline(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(selected, labels)
    warnings_seen = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    classifier = cast(LogisticRegression, pipeline.named_steps["logistic"])
    iterations = tuple(int(value) for value in classifier.n_iter_)
    converged = not warnings_seen and all(iteration < config.max_iter for iteration in iterations)
    if not converged:
        raise GraphModelTrainingError(
            f"graph classifier failed to converge within {config.max_iter} iterations"
        )
    classes = tuple(str(value) for value in classifier.classes_)
    if frozenset(classes) != frozenset(CLASS_ORDER):
        raise GraphModelTrainingError(f"graph classifier fitted invalid classes: {classes}")
    return FittedGraphClassifier(
        pipeline=pipeline,
        config=config,
        context=context,
        feature_columns=feature_columns,
        graph_feature_columns=graph_columns,
        sklearn_classes=classes,
        iterations=iterations,
        converged=True,
    )


def predict_graph_classifier_proba(
    model: FittedGraphClassifier,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return finite probabilities in explicit H, D, A order."""

    selected = _selected_numeric(frame, model.feature_columns)
    raw = np.asarray(model.pipeline.predict_proba(selected), dtype=np.float64)
    try:
        order = [model.sklearn_classes.index(label) for label in CLASS_ORDER]
    except ValueError as exc:
        raise GraphModelTrainingError("graph classifier does not contain H/D/A") from exc
    probabilities = validate_probabilities(raw[:, order])
    return pd.DataFrame(
        probabilities,
        index=frame.index.copy(),
        columns=PROBABILITY_COLUMNS,
    )
