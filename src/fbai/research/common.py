"""Immutable, aggregate-only records shared by experimental candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fbai.core.metrics import CLASS_ORDER, weighted_fold_summary
from fbai.core.splits import FoldRole
from fbai.evaluation.report import MetricRecord


@dataclass(frozen=True, slots=True)
class ResearchGate:
    """Predefined development gate for a candidate model."""

    minimum_improvement_log_loss: float = 0.005
    minimum_improving_folds: int = 2
    development_fold_count: int = 3

    def __post_init__(self) -> None:
        if self.minimum_improvement_log_loss <= 0.0:
            raise ValueError("minimum_improvement_log_loss must be positive")
        if self.development_fold_count < 1:
            raise ValueError("development_fold_count must be positive")
        if not 1 <= self.minimum_improving_folds <= self.development_fold_count:
            raise ValueError("minimum_improving_folds must be within the development fold count")

    def evaluate(
        self,
        *,
        lr52_log_loss: float,
        candidate_log_loss: float,
        fold_improvements: tuple[float, ...],
        historical_final_improvement_log_loss: float,
        passing_disposition: str = "MATCH2VEC_CANDIDATE",
        failing_disposition: str = "MATCH2VEC_REJECTED_FOR_NOW",
    ) -> ResearchGateResult:
        """Evaluate the frozen gate with positive values meaning candidate improvement."""

        if len(fold_improvements) != self.development_fold_count:
            raise ValueError(
                f"Gate requires exactly {self.development_fold_count} development folds"
            )
        if not passing_disposition or not failing_disposition:
            raise ValueError("Gate dispositions must be non-empty")
        improvement = lr52_log_loss - candidate_log_loss
        improving_folds = sum(value > 0.0 for value in fold_improvements)
        passed = (
            improvement >= self.minimum_improvement_log_loss
            and improving_folds >= self.minimum_improving_folds
        )
        disposition = passing_disposition if passed else failing_disposition
        return ResearchGateResult(
            definition=self,
            candidate_improvement_log_loss=improvement,
            improving_fold_count=improving_folds,
            historical_final_improvement_log_loss=historical_final_improvement_log_loss,
            passed=passed,
            disposition=disposition,
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "primary_metric": "development_sample_count_weighted_log_loss",
            "improvement_sign_convention": "lr52_log_loss_minus_candidate_log_loss",
            "minimum_improvement_log_loss": self.minimum_improvement_log_loss,
            "minimum_improving_folds": self.minimum_improving_folds,
            "development_fold_count": self.development_fold_count,
        }


@dataclass(frozen=True, slots=True)
class ResearchGateResult:
    """Observed values and disposition for a predefined gate."""

    definition: ResearchGate
    candidate_improvement_log_loss: float
    improving_fold_count: int
    historical_final_improvement_log_loss: float
    passed: bool
    disposition: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.to_dict(),
            "observed": {
                "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
                "improving_fold_count": self.improving_fold_count,
                "historical_final_improvement_log_loss": (
                    self.historical_final_improvement_log_loss
                ),
            },
            "passed": self.passed,
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class CandidateFoldEvaluation:
    """Aggregate-only LR52/candidate comparison for one identical-row fold."""

    fold_name: str
    test_year: int
    role: FoldRole
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    train_rows: int
    representation_fit_rows: int
    representation_validation_rows: int
    test_rows: int
    vocabulary_team_tokens: int
    vocabulary_unknown_tokens: int
    vocabulary_total_tokens: int
    oov_home_occurrences: int
    oov_away_occurrences: int
    training_epochs: int
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
            "representation_fit_rows": self.representation_fit_rows,
            "representation_validation_rows": self.representation_validation_rows,
            "test_rows": self.test_rows,
            "vocabulary": {
                "team_tokens": self.vocabulary_team_tokens,
                "unknown_tokens": self.vocabulary_unknown_tokens,
                "total_tokens": self.vocabulary_total_tokens,
            },
            "oov_occurrences": {
                "home": self.oov_home_occurrences,
                "away": self.oov_away_occurrences,
                "total": self.oov_home_occurrences + self.oov_away_occurrences,
            },
            "training_epochs": self.training_epochs,
            "identical_test_rows": self.identical_test_rows,
            "lr52": self.lr52.to_dict(),
            "candidate": self.candidate.to_dict(),
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
        }


@dataclass(frozen=True, slots=True)
class CandidateAggregate:
    """Sample-count-weighted LR52/candidate metrics for a named fold view."""

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


def aggregate_candidate_folds(
    name: str,
    folds: tuple[CandidateFoldEvaluation, ...],
) -> CandidateAggregate:
    """Aggregate candidate comparisons using test sample counts."""

    if not folds:
        raise ValueError(f"Aggregate {name} requires at least one fold")
    lr52 = MetricRecord.from_mapping(weighted_fold_summary([fold.lr52.to_dict() for fold in folds]))
    candidate = MetricRecord.from_mapping(
        weighted_fold_summary([fold.candidate.to_dict() for fold in folds])
    )
    return CandidateAggregate(
        name=name,
        fold_count=len(folds),
        lr52=lr52,
        candidate=candidate,
        candidate_improvement_log_loss=lr52.log_loss - candidate.log_loss,
    )


@dataclass(frozen=True, slots=True)
class CandidateComparisonReport:
    """Immutable report separating development, historical final, and diagnostic views."""

    schema_version: str
    experiment_id: str
    candidate_name: str
    candidate_configuration_id: str
    class_order: tuple[str, str, str]
    representation_feature_count: int
    representation_feature_names: tuple[str, ...]
    folds: tuple[CandidateFoldEvaluation, ...]
    development: CandidateAggregate
    historical_final: CandidateAggregate
    all_historical_diagnostic: CandidateAggregate
    success_gate: ResearchGateResult

    def __post_init__(self) -> None:
        if self.class_order != CLASS_ORDER:
            raise ValueError(f"class_order must be exactly {CLASS_ORDER}")
        if self.representation_feature_count != len(self.representation_feature_names):
            raise ValueError("representation feature count does not match its names")

    @property
    def disposition(self) -> str:
        return self.success_gate.disposition

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "candidate_name": self.candidate_name,
            "candidate_configuration_id": self.candidate_configuration_id,
            "class_order": list(self.class_order),
            "representation_feature_count": self.representation_feature_count,
            "representation_feature_names": list(self.representation_feature_names),
            "folds": [fold.to_dict() for fold in self.folds],
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "all_historical_diagnostic": self.all_historical_diagnostic.to_dict(),
            "success_gate": self.success_gate.to_dict(),
            "verdict": "passed" if self.success_gate.passed else "failed",
            "disposition": self.disposition,
        }
