"""Immutable, JSON-safe chronological LR52 evaluation records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from fbai.core.metrics import (
    CLASS_ORDER,
    FoldMetrics,
    WeightedFoldSummary,
    weighted_fold_summary,
)
from fbai.core.splits import FoldRole


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Probabilistic metrics for one fold or weighted aggregate."""

    n_samples: int
    log_loss: float
    brier_score: float
    ece: float

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float | int] | FoldMetrics | WeightedFoldSummary,
    ) -> Self:
        return cls(
            n_samples=int(values["n_samples"]),
            log_loss=float(values["log_loss"]),
            brier_score=float(values["brier_score"]),
            ece=float(values["ece"]),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_samples": self.n_samples,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "ece": self.ece,
        }


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    """Complete aggregate-only record for one chronological fold."""

    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    test_rows: int
    model_configuration_id: str
    preprocessing_fit_rows: int
    model_iterations: tuple[int, ...]
    model: MetricRecord
    uniform: MetricRecord
    training_prior: MetricRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_name": self.fold_name,
            "test_year": self.test_year,
            "role": self.role.value,
            "train_date_range": [self.train_start_date, self.train_end_date],
            "test_date_range": [self.test_start_date, self.test_end_date],
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "model_configuration_id": self.model_configuration_id,
            "preprocessing_fit_rows": self.preprocessing_fit_rows,
            "model_iterations": list(self.model_iterations),
            "model": self.model.to_dict(),
            "uniform": self.uniform.to_dict(),
            "training_prior": self.training_prior.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AggregateEvaluation:
    """Sample-count-weighted metrics for an explicitly named fold view."""

    name: str
    fold_count: int
    model: MetricRecord
    uniform: MetricRecord
    training_prior: MetricRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold_count": self.fold_count,
            "model": self.model.to_dict(),
            "uniform": self.uniform.to_dict(),
            "training_prior": self.training_prior.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Public LR52 report with development/final/diagnostic separation."""

    schema_version: str
    model_name: str
    model_configuration_id: str
    feature_count: int
    class_order: tuple[str, str, str]
    folds: tuple[FoldEvaluation, ...]
    development: AggregateEvaluation
    historical_final: AggregateEvaluation
    all_historical_diagnostic: AggregateEvaluation

    def to_dict(self) -> dict[str, Any]:
        """Return only JSON-safe aggregate and fold metadata."""

        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_configuration_id": self.model_configuration_id,
            "feature_count": self.feature_count,
            "class_order": list(self.class_order),
            "folds": [fold.to_dict() for fold in self.folds],
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
        }


def _aggregate(name: str, folds: tuple[FoldEvaluation, ...]) -> AggregateEvaluation:
    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")

    def summarize(attribute: str) -> MetricRecord:
        metrics = [getattr(fold, attribute).to_dict() for fold in folds]
        return MetricRecord.from_mapping(weighted_fold_summary(metrics))

    return AggregateEvaluation(
        name=name,
        fold_count=len(folds),
        model=summarize("model"),
        uniform=summarize("uniform"),
        training_prior=summarize("training_prior"),
    )


def build_evaluation_report(
    folds: tuple[FoldEvaluation, ...],
    *,
    model_configuration_id: str,
    feature_count: int,
) -> EvaluationReport:
    """Build separated weighted views from immutable fold records."""

    development = tuple(fold for fold in folds if fold.role is FoldRole.DEVELOPMENT)
    historical_final = tuple(fold for fold in folds if fold.role is FoldRole.HISTORICAL_FINAL)
    if len(historical_final) != 1:
        raise ValueError("Evaluation requires exactly one historically frozen final fold")
    return EvaluationReport(
        schema_version="1.0",
        model_name="LR52",
        model_configuration_id=model_configuration_id,
        feature_count=feature_count,
        class_order=CLASS_ORDER,
        folds=folds,
        development=_aggregate("development", development),
        historical_final=_aggregate("historical_final", historical_final),
        all_historical_diagnostic=_aggregate("all_historical_diagnostic", folds),
    )
