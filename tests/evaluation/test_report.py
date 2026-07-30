from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fbai.core.splits import FoldRole
from fbai.evaluation.report import (
    FoldEvaluation,
    MetricRecord,
    build_evaluation_report,
)


def metric(n: int, value: float) -> MetricRecord:
    return MetricRecord(n_samples=n, log_loss=value, brier_score=value, ece=value)


def fold(year: int, role: FoldRole, n: int, value: float) -> FoldEvaluation:
    return FoldEvaluation(
        fold_name=f"fold-{year}",
        test_year=year,
        role=role,
        train_start_date="2020-08-01",
        train_end_date=f"{year - 1}-05-30",
        test_start_date=f"{year}-08-01",
        test_end_date=f"{year + 1}-05-30",
        train_rows=100,
        test_rows=n,
        model_configuration_id="lr52-test",
        preprocessing_fit_rows=100,
        model_iterations=(12,),
        model=metric(n, value),
        uniform=metric(n, value + 1.0),
        training_prior=metric(n, value + 2.0),
    )


def test_report_structures_are_immutable_and_json_safe() -> None:
    report = build_evaluation_report(
        (
            fold(2022, FoldRole.DEVELOPMENT, 1, 1.0),
            fold(2023, FoldRole.DEVELOPMENT, 3, 3.0),
            fold(2025, FoldRole.HISTORICAL_FINAL, 2, 5.0),
        ),
        model_configuration_id="lr52-test",
        feature_count=52,
    )

    with pytest.raises(FrozenInstanceError):
        report.feature_count = 1  # type: ignore[misc]
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["class_order"] == ["H", "D", "A"]
    assert "C:\\" not in serialized
    assert "/home/" not in serialized
    assert payload["development"]["model"]["log_loss"] == pytest.approx(2.5)
    assert payload["historical_final"]["fold_count"] == 1
    assert payload["all_historical_diagnostic"]["name"] == "all_historical_diagnostic"


def test_numpy_metric_values_are_normalized_to_python_scalars() -> None:
    record = MetricRecord.from_mapping(
        {
            "n_samples": np.int64(2),
            "log_loss": np.float64(1.0),
            "brier_score": np.float64(0.5),
            "ece": np.float64(0.1),
        }
    )

    assert type(record.n_samples) is int
    assert type(record.log_loss) is float
    json.dumps(record.to_dict())
