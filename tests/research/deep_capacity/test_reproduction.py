from __future__ import annotations

import json
import re
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parents[3] / "research" / "deep_capacity" / "result.json"


def _result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_result_declares_exact_architectures_and_selection_roles() -> None:
    result = _result()

    assert result["schema_version"] == "1.0"
    assert result["experiment_id"] == "deep_capacity_lr52_historical_reproduction"
    assert result["status"] == "reproduced"
    assert result["class_order"] == ["H", "D", "A"]
    assert result["input_feature_count"] == 52
    assert [item["parameter_count"] for item in result["architectures"]] == [  # type: ignore[index]
        3587,
        15235,
    ]
    assert result["selection"]["development_only"]  # type: ignore[index]


def test_result_records_exact_gate_and_disposition() -> None:
    result = _result()
    gate = result["success_gate"]

    assert gate["definition"]["minimum_improvement_log_loss"] == 0.005  # type: ignore[index]
    assert gate["definition"]["minimum_improving_folds"] == 2  # type: ignore[index]
    assert not gate["passed"]  # type: ignore[index]
    assert result["verdict"] == "failed_gate"
    assert result["disposition"] == "DATA_CEILING_UPHELD"


def test_reference_comparison_is_explicit_and_within_tolerance() -> None:
    comparison = _result()["reference_comparison"]

    assert comparison["reproduction_passed"]  # type: ignore[index]
    tolerances = comparison["comparison_tolerances"]  # type: ignore[index]
    for candidate in comparison["absolute_differences"].values():  # type: ignore[index,union-attr]
        for differences in candidate["development_folds"].values():
            assert differences["log_loss"] <= tolerances["log_loss"]
            assert differences["brier_score"] <= tolerances["brier_score"]
            assert differences["ece"] <= tolerances["ece"]


def test_result_contains_only_aggregate_safe_metadata() -> None:
    text = RESULT_PATH.read_text(encoding="utf-8")
    result = _result()

    assert not re.search(r"[A-Za-z]:\\\\|/home/|/Users/", text)
    assert "HomeTeam" not in text
    assert "AwayTeam" not in text
    forbidden_keys = {
        "bankroll",
        "bets",
        "checkpoint",
        "odds",
        "paths",
        "predictions",
        "raw_data",
        "roi",
        "staking",
        "team_names",
        "tensors",
        "weights",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert forbidden_keys.isdisjoint(keys(result))
    assert text.endswith("\n")
