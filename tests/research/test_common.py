from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fbai.research.common import ResearchGate


def test_gate_uses_positive_candidate_improvement_sign() -> None:
    result = ResearchGate().evaluate(
        lr52_log_loss=1.0,
        candidate_log_loss=0.994,
        fold_improvements=(0.001, 0.002, -0.001),
        historical_final_improvement_log_loss=0.004,
    )

    assert result.candidate_improvement_log_loss == pytest.approx(0.006)
    assert result.improving_fold_count == 2
    assert result.passed


def test_gate_requires_both_predefined_criteria() -> None:
    too_small = ResearchGate().evaluate(
        lr52_log_loss=1.0,
        candidate_log_loss=0.996,
        fold_improvements=(0.001, 0.002, 0.003),
        historical_final_improvement_log_loss=0.0,
    )
    too_few_folds = ResearchGate().evaluate(
        lr52_log_loss=1.0,
        candidate_log_loss=0.99,
        fold_improvements=(0.001, -0.001, -0.002),
        historical_final_improvement_log_loss=0.0,
    )

    assert not too_small.passed
    assert not too_few_folds.passed
    assert too_small.disposition == "MATCH2VEC_REJECTED_FOR_NOW"


def test_gate_is_immutable_and_json_safe() -> None:
    gate = ResearchGate()

    with pytest.raises(FrozenInstanceError):
        gate.minimum_improving_folds = 1  # type: ignore[misc]
    assert gate.to_dict()["improvement_sign_convention"] == (
        "lr52_log_loss_minus_candidate_log_loss"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_improvement_log_loss": 0.0}, "positive"),
        ({"minimum_improving_folds": 0}, "within"),
        ({"minimum_improving_folds": 4}, "within"),
    ],
)
def test_invalid_gate_definitions_fail(kwargs: dict[str, float | int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ResearchGate(**kwargs)  # type: ignore[arg-type]
