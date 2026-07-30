"""Development-selected deep-capacity evaluation against identical-row LR52."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from fbai.core.metrics import (
    CLASS_ORDER,
    evaluate_predictions,
    weighted_fold_summary,
)
from fbai.core.splits import (
    ALL_TEST_YEARS,
    DEVELOPMENT_TEST_YEARS,
    STABLE_ORDER_KEY,
    FoldRole,
    expanding_folds,
    inner_time_split,
)
from fbai.evaluation.report import MetricRecord
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, fit_lr52, predict_lr52_proba
from fbai.models.preprocessing import select_lr52_features
from fbai.research.common import ResearchGate, ResearchGateResult
from fbai.research.deep_capacity.config import (
    AUTHORITATIVE_CONFIGS,
    DeepCapacityConfig,
)
from fbai.research.deep_capacity.model import DeepCapacityModel

EXPERIMENT_ID = "deep_capacity_lr52_historical_reproduction"
CANDIDATE_FAMILY = "internal_mlp_lr52"


class DeepCapacityEvaluationError(ValueError):
    """Raised when the frozen deep-capacity protocol is violated."""


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date().isoformat())


def _metrics(
    labels: pd.Series,
    probabilities: np.ndarray | pd.DataFrame,
    *,
    n_bins: int,
) -> MetricRecord:
    values = (
        probabilities.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if isinstance(probabilities, pd.DataFrame)
        else np.asarray(probabilities, dtype=np.float64)
    )
    return MetricRecord.from_mapping(
        evaluate_predictions(labels.astype(str).tolist(), values, n_bins=n_bins)
    )


def _ordered_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_table, pd.DataFrame):
        raise DeepCapacityEvaluationError("deep-capacity input must be a pandas DataFrame")
    ordered = (
        feature_table.sort_values(list(STABLE_ORDER_KEY), kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )
    validate_feature_table(ordered, expected_row_count=len(feature_table))
    return ordered


def _numeric_inputs(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit source median imputation and scaling on inner-fit rows only."""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    fit_values = scaler.fit_transform(imputer.fit_transform(select_lr52_features(fit)))
    validation_values = scaler.transform(imputer.transform(select_lr52_features(validation)))
    test_values = scaler.transform(imputer.transform(select_lr52_features(test)))
    arrays = (
        np.asarray(fit_values, dtype=np.float32),
        np.asarray(validation_values, dtype=np.float32),
        np.asarray(test_values, dtype=np.float32),
    )
    if any(values.shape[1] != len(FEATURE_COLUMNS) for values in arrays):
        raise DeepCapacityEvaluationError(
            "deep-capacity preprocessing must preserve all 52 LR52 features"
        )
    if any(not np.isfinite(values).all() for values in arrays):
        raise DeepCapacityEvaluationError("deep-capacity preprocessing produced non-finite inputs")
    return arrays


@dataclass(frozen=True, slots=True)
class DeepCapacityFoldEvaluation:
    """Aggregate-only metrics and training boundaries for one architecture/fold."""

    architecture_name: str
    architecture_identifier: str
    parameter_count: int
    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    fit_start_date: str
    fit_end_date: str
    validation_start_date: str
    validation_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    fit_rows: int
    validation_rows: int
    test_rows: int
    preprocessing_fit_rows: int
    training_epochs: int
    best_validation_loss: float
    same_date_isolation: bool
    identical_test_rows: bool
    lr52: MetricRecord
    candidate: MetricRecord
    candidate_improvement_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_name": self.architecture_name,
            "architecture_identifier": self.architecture_identifier,
            "parameter_count": self.parameter_count,
            "fold_name": self.fold_name,
            "test_year": self.test_year,
            "role": self.role.value,
            "train_date_range": [self.train_start_date, self.train_end_date],
            "fit_date_range": [self.fit_start_date, self.fit_end_date],
            "validation_date_range": [
                self.validation_start_date,
                self.validation_end_date,
            ],
            "test_date_range": [self.test_start_date, self.test_end_date],
            "train_rows": self.train_rows,
            "fit_rows": self.fit_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "preprocessing_fit_rows": self.preprocessing_fit_rows,
            "training_epochs": self.training_epochs,
            "best_validation_loss": self.best_validation_loss,
            "same_date_isolation": self.same_date_isolation,
            "identical_test_rows": self.identical_test_rows,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


@dataclass(frozen=True, slots=True)
class DeepCapacityAggregate:
    """Sample-count-weighted LR52/candidate metrics for a named view."""

    name: str
    fold_count: int
    lr52: MetricRecord
    candidate: MetricRecord
    candidate_improvement_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold_count": self.fold_count,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


def aggregate_deep_capacity_folds(
    name: str,
    folds: tuple[DeepCapacityFoldEvaluation, ...],
) -> DeepCapacityAggregate:
    """Aggregate fold metrics by test row count."""

    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")
    lr52 = MetricRecord.from_mapping(weighted_fold_summary([fold.lr52.to_dict() for fold in folds]))
    candidate = MetricRecord.from_mapping(
        weighted_fold_summary([fold.candidate.to_dict() for fold in folds])
    )
    return DeepCapacityAggregate(
        name=name,
        fold_count=len(folds),
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


@dataclass(frozen=True, slots=True)
class DeepCapacityCandidateEvaluation:
    """Development-only evaluation for one source-defined architecture."""

    configuration: DeepCapacityConfig
    development_folds: tuple[DeepCapacityFoldEvaluation, ...]
    development: DeepCapacityAggregate

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.to_dict(),
            "development_folds": [fold.to_dict() for fold in self.development_folds],
            "development": self.development.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DeepCapacityComparisonReport:
    """Immutable architecture-selection and historical-final report."""

    schema_version: str
    experiment_id: str
    candidate_family: str
    class_order: tuple[str, str, str]
    input_feature_count: int
    configurations: tuple[DeepCapacityConfig, ...]
    candidates: tuple[DeepCapacityCandidateEvaluation, ...]
    selected_architecture: str
    selection_metric: str
    selection_used_development_only: bool
    selected_final_fold: DeepCapacityFoldEvaluation
    historical_final: DeepCapacityAggregate
    all_historical_diagnostic: DeepCapacityAggregate
    success_gate: ResearchGateResult
    disposition: str

    def __post_init__(self) -> None:
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
        if self.input_feature_count != len(FEATURE_COLUMNS):
            raise ValueError("deep-capacity input must preserve the exact LR52 tuple")
        names = tuple(configuration.name for configuration in self.configurations)
        if self.selected_architecture not in names:
            raise ValueError("selected deep-capacity architecture is not configured")
        if not self.selection_used_development_only:
            raise ValueError("deep-capacity selection must use development folds only")
        if self.selected_final_fold.architecture_name != self.selected_architecture:
            raise ValueError("historical-final fold must use only the selected architecture")

    @property
    def selected_candidate(self) -> DeepCapacityCandidateEvaluation:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.configuration.name == self.selected_architecture
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "candidate_family": self.candidate_family,
            "class_order": list(self.class_order),
            "input_feature_count": self.input_feature_count,
            "configurations": [configuration.to_dict() for configuration in self.configurations],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection": {
                "architecture": self.selected_architecture,
                "metric": self.selection_metric,
                "development_only": self.selection_used_development_only,
            },
            "selected_final_fold": self.selected_final_fold.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
            "success_gate": self.success_gate.to_dict(),
            "verdict": "passed" if self.success_gate.passed else "failed",
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class _FoldInputs:
    fold_name: str
    test_year: int
    role: FoldRole
    train: pd.DataFrame
    test: pd.DataFrame
    lr52: MetricRecord


def _evaluate_architecture_fold(
    inputs: _FoldInputs,
    configuration: DeepCapacityConfig,
    *,
    n_bins: int,
) -> DeepCapacityFoldEvaluation:
    train = inputs.train
    test = inputs.test
    fit_indices, validation_indices = inner_time_split(
        train,
        validation_fraction=configuration.validation_fraction,
    )
    fit = train.loc[list(fit_indices)].copy(deep=True)
    validation = train.loc[list(validation_indices)].copy(deep=True)
    same_date_isolation = fit["MatchDate"].max() < validation["MatchDate"].min()
    if not same_date_isolation:
        raise DeepCapacityEvaluationError(
            f"Fold {inputs.fold_name} splits a date across fit and validation"
        )
    fit_numeric, validation_numeric, test_numeric = _numeric_inputs(
        fit,
        validation,
        test,
    )
    model = DeepCapacityModel(configuration).fit(
        fit_numeric,
        fit["target_1x2"].astype(str).tolist(),
        validation_numeric,
        validation["target_1x2"].astype(str).tolist(),
    )
    probabilities = model.predict_proba(test_numeric)
    identical_test_rows = len(probabilities) == len(test)
    if not identical_test_rows:
        raise DeepCapacityEvaluationError(
            "deep-capacity candidate and LR52 test keys are not identical"
        )
    candidate = _metrics(test["target_1x2"], probabilities, n_bins=n_bins)
    return DeepCapacityFoldEvaluation(
        architecture_name=configuration.name,
        architecture_identifier=configuration.identifier,
        parameter_count=model.parameter_count,
        fold_name=inputs.fold_name,
        test_year=inputs.test_year,
        role=inputs.role,
        train_start_date=_date_text(train["MatchDate"].min()),
        train_end_date=_date_text(train["MatchDate"].max()),
        fit_start_date=_date_text(fit["MatchDate"].min()),
        fit_end_date=_date_text(fit["MatchDate"].max()),
        validation_start_date=_date_text(validation["MatchDate"].min()),
        validation_end_date=_date_text(validation["MatchDate"].max()),
        test_start_date=_date_text(test["MatchDate"].min()),
        test_end_date=_date_text(test["MatchDate"].max()),
        train_rows=len(train),
        fit_rows=len(fit),
        validation_rows=len(validation),
        test_rows=len(test),
        preprocessing_fit_rows=len(fit),
        training_epochs=model.epochs_trained,
        best_validation_loss=model.best_validation_loss,
        same_date_isolation=True,
        identical_test_rows=True,
        lr52=inputs.lr52,
        candidate=candidate,
        candidate_improvement_log_loss=inputs.lr52.log_loss - candidate.log_loss,
    )


def evaluate_deep_capacity(
    feature_table: pd.DataFrame,
    *,
    configurations: tuple[DeepCapacityConfig, ...] = AUTHORITATIVE_CONFIGS,
    test_years: tuple[int, ...] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> DeepCapacityComparisonReport:
    """Select source MLP capacity on development and evaluate final once."""

    if tuple(test_years) != ALL_TEST_YEARS:
        raise ValueError(
            "The predefined deep-capacity audit requires the complete 2022-2025 protocol"
        )
    if not configurations:
        raise ValueError("deep-capacity evaluation requires at least one architecture")
    names = tuple(configuration.name for configuration in configurations)
    if len(names) != len(set(names)):
        raise ValueError("deep-capacity architecture names must be unique")
    original = feature_table.copy(deep=True)
    ordered = _ordered_feature_table(feature_table)
    fold_inputs: list[_FoldInputs] = []
    for fold in expanding_folds(ordered, test_years=test_years):
        train = ordered.loc[list(fold.train_idx)].copy(deep=True)
        test = ordered.loc[list(fold.test_idx)].copy(deep=True)
        if train["MatchDate"].max() >= test["MatchDate"].min():
            raise DeepCapacityEvaluationError(
                f"Fold {fold.name} violates strict train-before-test chronology"
            )
        lr52_probabilities = predict_lr52_proba(fit_lr52(train), test)
        fold_inputs.append(
            _FoldInputs(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train=train,
                test=test,
                lr52=_metrics(test["target_1x2"], lr52_probabilities, n_bins=n_bins),
            )
        )

    development_inputs = tuple(
        inputs for inputs in fold_inputs if inputs.test_year in DEVELOPMENT_TEST_YEARS
    )
    final_inputs = tuple(
        inputs for inputs in fold_inputs if inputs.role is FoldRole.HISTORICAL_FINAL
    )
    if len(development_inputs) != 3 or len(final_inputs) != 1:
        raise DeepCapacityEvaluationError(
            "deep-capacity evaluation requires three development and one final fold"
        )

    candidates: list[DeepCapacityCandidateEvaluation] = []
    for configuration in configurations:
        folds = tuple(
            _evaluate_architecture_fold(inputs, configuration, n_bins=n_bins)
            for inputs in development_inputs
        )
        candidates.append(
            DeepCapacityCandidateEvaluation(
                configuration=configuration,
                development_folds=folds,
                development=aggregate_deep_capacity_folds("development", folds),
            )
        )
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.development.candidate.log_loss,
            candidate.configuration.name,
        ),
    )
    selected_final = _evaluate_architecture_fold(
        final_inputs[0],
        selected.configuration,
        n_bins=n_bins,
    )
    historical_final = aggregate_deep_capacity_folds(
        "historical_final",
        (selected_final,),
    )
    all_historical = aggregate_deep_capacity_folds(
        "all_historical_diagnostic",
        (*selected.development_folds, selected_final),
    )
    gate = ResearchGate().evaluate(
        lr52_log_loss=selected.development.lr52.log_loss,
        candidate_log_loss=selected.development.candidate.log_loss,
        fold_improvements=tuple(
            fold.candidate_improvement_log_loss for fold in selected.development_folds
        ),
        historical_final_improvement_log_loss=(historical_final.candidate_improvement_log_loss),
        passing_disposition="DEEP_MODEL_CANDIDATE",
        failing_disposition="DATA_CEILING_UPHELD",
    )
    report = DeepCapacityComparisonReport(
        schema_version="1.0",
        experiment_id=EXPERIMENT_ID,
        candidate_family=CANDIDATE_FAMILY,
        class_order=CLASS_ORDER,
        input_feature_count=len(FEATURE_COLUMNS),
        configurations=configurations,
        candidates=tuple(candidates),
        selected_architecture=selected.configuration.name,
        selection_metric="development_sample_count_weighted_log_loss",
        selection_used_development_only=True,
        selected_final_fold=selected_final,
        historical_final=historical_final,
        all_historical_diagnostic=all_historical,
        success_gate=gate,
        disposition=gate.disposition,
    )
    pd.testing.assert_frame_equal(feature_table, original)
    return report
