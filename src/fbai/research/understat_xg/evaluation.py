"""Chronological LR52-plus-Understat comparison on identical covered rows."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fbai.core.leakage import assert_model_inputs_safe
from fbai.core.metrics import CLASS_ORDER, evaluate_predictions, validate_probabilities
from fbai.core.splits import (
    ALL_TEST_YEARS,
    DEVELOPMENT_TEST_YEARS,
    STABLE_ORDER_KEY,
    FoldRole,
    expanding_folds,
)
from fbai.data.schema import NATURAL_KEY, validate_canonical_frame
from fbai.evaluation.report import MetricRecord
from fbai.features.checks import validate_feature_table
from fbai.features.schema import FEATURE_COLUMNS, FEATURE_TABLE_COLUMNS
from fbai.models.logistic import PROBABILITY_COLUMNS, TARGET_COLUMN, fit_lr52, predict_lr52_proba
from fbai.research.common import ResearchGate, ResearchGateResult
from fbai.research.understat_xg.alignment import (
    UnderstatXGAlignmentReport,
    align_understat_xg,
)
from fbai.research.understat_xg.features import (
    UNDERSTAT_XG_FEATURE_COLUMNS,
    build_understat_xg_feature_table,
)

EXPERIMENT_ID = "understat_xg_lr68_historical_reproduction"
CANDIDATE_NAME = "LR52 + 16 historical Understat xG features"


class UnderstatXGEvaluationError(ValueError):
    """Raised when candidate evaluation violates the frozen protocol."""


class UnderstatXGConvergenceError(RuntimeError):
    """Raised when the source-defined candidate does not converge."""


@dataclass(frozen=True, slots=True)
class UnderstatXGConfig:
    """Exact effective configuration of the authoritative primary experiment."""

    rolling_windows: tuple[int, int] = (5, 10)
    history_season_scope: str = "SeasonStartYear"
    history_division_scope: str = "Division"
    partial_windows: bool = True
    same_date_batching: bool = True
    candidate: str = "LogisticRegression"
    candidate_base_feature_count: int = 52
    candidate_xg_feature_count: int = 16
    imputation: str = "median"
    scaling: str = "standard"
    logistic_solver: str = "lbfgs"
    logistic_penalty: str = "l2"
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    logistic_fit_intercept: bool = True
    logistic_class_weight: str | None = None
    logistic_random_state: int = 42
    logistic_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.rolling_windows != (5, 10):
            raise ValueError("Understat xG rolling windows are fixed to 5 and 10")
        if (
            self.history_season_scope != "SeasonStartYear"
            or self.history_division_scope != "Division"
        ):
            raise ValueError("Understat xG history must reset by season within division")
        if not self.partial_windows or not self.same_date_batching:
            raise ValueError("Understat xG requires partial windows and same-date batching")
        if self.candidate != "LogisticRegression":
            raise ValueError("Understat xG candidate is fixed to LogisticRegression")
        if self.candidate_base_feature_count != 52 or self.candidate_xg_feature_count != 16:
            raise ValueError("Understat xG candidate is fixed to 52 plus 16 features")
        if self.imputation != "median" or self.scaling != "standard":
            raise ValueError("Understat xG preprocessing is fixed to median then standard scaling")
        if self.logistic_solver != "lbfgs" or self.logistic_penalty != "l2":
            raise ValueError("Understat xG Logistic Regression solver/penalty are frozen")
        if self.logistic_c != 1.0 or self.logistic_max_iter != 2000:
            raise ValueError("Understat xG Logistic Regression C/iteration limit are frozen")
        if not self.logistic_fit_intercept or self.logistic_class_weight is not None:
            raise ValueError("Understat xG Logistic Regression intercept/class weight are frozen")
        if self.logistic_random_state != 42 or self.logistic_tolerance <= 0.0:
            raise ValueError("Understat xG Logistic Regression seed/tolerance are frozen")

    @property
    def identifier(self) -> str:
        return "understat_w5w10_lr68_median_standardized_lbfgs_l2_c1_iter2000_seed42"

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = asdict(self)
        values["rolling_windows"] = list(self.rolling_windows)
        values["identifier"] = self.identifier
        return values


@dataclass(frozen=True, slots=True)
class FittedUnderstatXGCandidate:
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    sklearn_classes: tuple[str, ...]
    iterations: tuple[int, ...]


def candidate_feature_columns() -> tuple[str, ...]:
    """Return the explicit experimental 68-feature tuple."""

    return (*FEATURE_COLUMNS, *UNDERSTAT_XG_FEATURE_COLUMNS)


def _select_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = candidate_feature_columns()
    expected = set(FEATURE_TABLE_COLUMNS).union(columns)
    missing = [column for column in columns if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing:
        raise UnderstatXGEvaluationError(f"Missing Understat candidate features: {missing}")
    if extra:
        raise UnderstatXGEvaluationError(f"Unexpected Understat candidate columns: {extra}")
    assert_model_inputs_safe(columns, approved_pre_features=columns)
    selected = frame.loc[:, list(columns)].copy(deep=True)
    for column in columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    if np.isinf(selected.to_numpy(dtype=np.float64, na_value=np.nan)).any():
        raise UnderstatXGEvaluationError("Understat candidate inputs contain infinite values")
    return selected


def _fit_candidate(
    train_frame: pd.DataFrame,
    config: UnderstatXGConfig,
) -> FittedUnderstatXGCandidate:
    selected = _select_candidate_features(train_frame)
    if TARGET_COLUMN not in train_frame.columns or train_frame[TARGET_COLUMN].isna().any():
        raise UnderstatXGEvaluationError(f"Training frame requires complete {TARGET_COLUMN}")
    targets = train_frame[TARGET_COLUMN].astype(str)
    if frozenset(targets) != frozenset(CLASS_ORDER):
        raise UnderstatXGEvaluationError("Understat candidate training requires all H/D/A classes")
    classifier = LogisticRegression(
        C=config.logistic_c,
        solver=config.logistic_solver,
        max_iter=config.logistic_max_iter,
        fit_intercept=config.logistic_fit_intercept,
        class_weight=config.logistic_class_weight,
        random_state=config.logistic_random_state,
        tol=config.logistic_tolerance,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config.imputation)),
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
    if convergence_warnings or any(value >= config.logistic_max_iter for value in iterations):
        raise UnderstatXGConvergenceError(
            f"Understat xG candidate failed within {config.logistic_max_iter} iterations"
        )
    return FittedUnderstatXGCandidate(
        pipeline=pipeline,
        feature_columns=candidate_feature_columns(),
        sklearn_classes=tuple(str(value) for value in classifier.classes_),
        iterations=iterations,
    )


def _predict_candidate(
    fitted: FittedUnderstatXGCandidate,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if fitted.feature_columns != candidate_feature_columns():
        raise UnderstatXGEvaluationError("fitted Understat candidate feature tuple is invalid")
    raw = np.asarray(
        fitted.pipeline.predict_proba(_select_candidate_features(frame)),
        dtype=np.float64,
    )
    order = [fitted.sklearn_classes.index(label) for label in CLASS_ORDER]
    probabilities = validate_probabilities(raw[:, order])
    return pd.DataFrame(probabilities, index=frame.index.copy(), columns=PROBABILITY_COLUMNS)


def _metrics(
    labels: pd.Series,
    probabilities: pd.DataFrame,
    *,
    n_bins: int,
) -> MetricRecord:
    return MetricRecord.from_mapping(
        evaluate_predictions(
            labels.astype(str).tolist(),
            probabilities.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64),
            n_bins=n_bins,
        )
    )


def _date_text(value: object) -> str:
    return str(pd.Timestamp(value).date().isoformat())


@dataclass(frozen=True, slots=True)
class UnderstatXGFoldEvaluation:
    """Aggregate-only comparison for one independently fitted fold."""

    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    test_rows: int
    xg_feature_coverage: float
    candidate_preprocessing_fit_rows: int
    candidate_iterations: tuple[int, ...]
    identical_test_rows: bool
    lr52: MetricRecord
    candidate: MetricRecord
    candidate_improvement_log_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "test_year": self.test_year,
            "role": self.role.value,
            "train_date_range": [self.train_start_date, self.train_end_date],
            "test_date_range": [self.test_start_date, self.test_end_date],
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "xg_feature_coverage": self.xg_feature_coverage,
            "candidate_preprocessing_fit_rows": self.candidate_preprocessing_fit_rows,
            "candidate_iterations": list(self.candidate_iterations),
            "identical_test_rows": self.identical_test_rows,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


@dataclass(frozen=True, slots=True)
class UnderstatXGAggregate:
    """Sample-count-weighted metrics for a named fold view."""

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


def aggregate_understat_xg_folds(
    name: str,
    folds: tuple[UnderstatXGFoldEvaluation, ...],
) -> UnderstatXGAggregate:
    """Aggregate with test sample counts, never an unweighted fold mean."""

    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")
    total = sum(fold.test_rows for fold in folds)

    def aggregate(attribute: str) -> MetricRecord:
        metrics: list[MetricRecord] = [getattr(fold, attribute) for fold in folds]
        return MetricRecord(
            n_samples=total,
            log_loss=sum(item.log_loss * item.n_samples for item in metrics) / total,
            brier_score=sum(item.brier_score * item.n_samples for item in metrics) / total,
            ece=sum(item.ece * item.n_samples for item in metrics) / total,
        )

    lr52 = aggregate("lr52")
    candidate = aggregate("candidate")
    return UnderstatXGAggregate(
        name=name,
        fold_count=len(folds),
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


@dataclass(frozen=True, slots=True)
class UnderstatXGComparisonReport:
    """Immutable aggregate-only report for the external xG candidate."""

    schema_version: str
    experiment_id: str
    candidate_name: str
    configuration: UnderstatXGConfig
    class_order: tuple[str, str, str]
    base_feature_count: int
    xg_feature_names: tuple[str, ...]
    alignment: UnderstatXGAlignmentReport
    folds: tuple[UnderstatXGFoldEvaluation, ...]
    development: UnderstatXGAggregate
    historical_final: UnderstatXGAggregate
    all_historical_diagnostic: UnderstatXGAggregate
    success_gate: ResearchGateResult

    def __post_init__(self) -> None:
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
        if self.base_feature_count != len(FEATURE_COLUMNS):
            raise ValueError("base feature count must preserve the exact LR52 tuple")
        if self.xg_feature_names != UNDERSTAT_XG_FEATURE_COLUMNS:
            raise ValueError("Understat xG feature names must match the explicit contract")

    @property
    def disposition(self) -> str:
        return self.success_gate.disposition

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "candidate_name": self.candidate_name,
            "candidate_configuration": self.configuration.to_dict(),
            "class_order": list(self.class_order),
            "base_feature_count": self.base_feature_count,
            "xg_feature_count": len(self.xg_feature_names),
            "xg_feature_names": list(self.xg_feature_names),
            "alignment": self.alignment.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
            "success_gate": self.success_gate.to_dict(),
            "verdict": "passed" if self.success_gate.passed else "failed",
            "disposition": self.disposition,
        }


def _aligned_inputs(
    feature_table: pd.DataFrame,
    canonical_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_feature_table(feature_table, expected_row_count=len(feature_table))
    validate_canonical_frame(canonical_matches)
    ordered_features = feature_table.sort_values(
        list(STABLE_ORDER_KEY), kind="mergesort"
    ).reset_index(drop=True)
    ordered_canonical = canonical_matches.sort_values(
        list(STABLE_ORDER_KEY), kind="mergesort"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            ordered_features.loc[:, list(NATURAL_KEY)],
            ordered_canonical.loc[:, list(NATURAL_KEY)],
            check_dtype=False,
        )
    except AssertionError as exc:
        raise UnderstatXGEvaluationError(
            "Feature and canonical tables must contain identical natural keys"
        ) from exc
    return ordered_features, ordered_canonical


def evaluate_understat_xg_candidate(
    feature_table: pd.DataFrame,
    canonical_matches: pd.DataFrame,
    external_xg: pd.DataFrame,
    *,
    config: UnderstatXGConfig | None = None,
    test_years: tuple[int, ...] = ALL_TEST_YEARS,
    n_bins: int = 10,
) -> UnderstatXGComparisonReport:
    """Re-fit and compare the authoritative primary Understat candidate."""

    if tuple(test_years) != ALL_TEST_YEARS:
        raise ValueError(
            "The predefined Understat xG gate requires the complete 2022-2025 protocol"
        )
    original_features = feature_table.copy(deep=True)
    original_canonical = canonical_matches.copy(deep=True)
    original_external = external_xg.copy(deep=True)
    ordered, canonical = _aligned_inputs(feature_table, canonical_matches)
    aligned = align_understat_xg(canonical, external_xg)
    covered = set(aligned.report.covered_divisions)
    ordered = ordered.loc[ordered["Division"].isin(covered)].reset_index(drop=True)
    canonical = canonical.loc[canonical["Division"].isin(covered)].reset_index(drop=True)
    xg_features = build_understat_xg_feature_table(ordered, aligned.matches)
    augmented = ordered.merge(
        xg_features,
        on=list(NATURAL_KEY),
        how="left",
        sort=False,
        validate="one_to_one",
    )
    resolved = config or UnderstatXGConfig()
    records: list[UnderstatXGFoldEvaluation] = []

    for fold in expanding_folds(ordered, test_years=test_years):
        train = ordered.loc[list(fold.train_idx)].copy(deep=True)
        test = ordered.loc[list(fold.test_idx)].copy(deep=True)
        augmented_train = augmented.loc[list(fold.train_idx)].copy(deep=True)
        augmented_test = augmented.loc[list(fold.test_idx)].copy(deep=True)
        if train["MatchDate"].max() >= test["MatchDate"].min():
            raise UnderstatXGEvaluationError(
                f"Fold {fold.name} violates strict train-before-test chronology"
            )

        lr52_probabilities = predict_lr52_proba(fit_lr52(train), test)
        candidate_model = _fit_candidate(augmented_train, resolved)
        candidate_probabilities = _predict_candidate(candidate_model, augmented_test)
        identical_test_rows = len(lr52_probabilities) == len(candidate_probabilities) == len(
            test
        ) and test.loc[:, list(NATURAL_KEY)].reset_index(drop=True).equals(
            augmented_test.loc[:, list(NATURAL_KEY)].reset_index(drop=True)
        )
        if not identical_test_rows:
            raise UnderstatXGEvaluationError(
                "Understat candidate and LR52 test keys are not identical"
            )
        labels = test[TARGET_COLUMN]
        lr52_metrics = _metrics(labels, lr52_probabilities, n_bins=n_bins)
        candidate_metrics = _metrics(labels, candidate_probabilities, n_bins=n_bins)
        records.append(
            UnderstatXGFoldEvaluation(
                fold_name=fold.name,
                test_year=fold.test_year,
                role=fold.role,
                train_start_date=_date_text(train["MatchDate"].min()),
                train_end_date=_date_text(train["MatchDate"].max()),
                test_start_date=_date_text(test["MatchDate"].min()),
                test_end_date=_date_text(test["MatchDate"].max()),
                train_rows=len(train),
                test_rows=len(test),
                xg_feature_coverage=float(
                    augmented_test.loc[:, list(UNDERSTAT_XG_FEATURE_COLUMNS)]
                    .notna()
                    .any(axis=1)
                    .mean()
                ),
                candidate_preprocessing_fit_rows=len(augmented_train),
                candidate_iterations=candidate_model.iterations,
                identical_test_rows=True,
                lr52=lr52_metrics,
                candidate=candidate_metrics,
                candidate_improvement_log_loss=(lr52_metrics.log_loss - candidate_metrics.log_loss),
            )
        )

    folds = tuple(records)
    development_folds = tuple(fold for fold in folds if fold.test_year in DEVELOPMENT_TEST_YEARS)
    final_folds = tuple(fold for fold in folds if fold.role is FoldRole.HISTORICAL_FINAL)
    if len(development_folds) != 3 or len(final_folds) != 1:
        raise UnderstatXGEvaluationError(
            "Understat evaluation requires three development and one final fold"
        )
    development = aggregate_understat_xg_folds("development", development_folds)
    historical_final = aggregate_understat_xg_folds("historical_final", final_folds)
    all_historical = aggregate_understat_xg_folds("all_historical_diagnostic", folds)
    gate = ResearchGate().evaluate(
        lr52_log_loss=development.lr52.log_loss,
        candidate_log_loss=development.candidate.log_loss,
        fold_improvements=tuple(fold.candidate_improvement_log_loss for fold in development_folds),
        historical_final_improvement_log_loss=(historical_final.candidate_improvement_log_loss),
        passing_disposition="XG_SIGNAL_CANDIDATE",
        failing_disposition="XG_SIGNAL_REJECTED_FOR_NOW",
    )
    report = UnderstatXGComparisonReport(
        schema_version="1.0",
        experiment_id=EXPERIMENT_ID,
        candidate_name=CANDIDATE_NAME,
        configuration=resolved,
        class_order=CLASS_ORDER,
        base_feature_count=len(FEATURE_COLUMNS),
        xg_feature_names=UNDERSTAT_XG_FEATURE_COLUMNS,
        alignment=aligned.report,
        folds=folds,
        development=development,
        historical_final=historical_final,
        all_historical_diagnostic=all_historical,
        success_gate=gate,
    )
    pd.testing.assert_frame_equal(feature_table, original_features)
    pd.testing.assert_frame_equal(canonical_matches, original_canonical)
    pd.testing.assert_frame_equal(external_xg, original_external)
    return report
