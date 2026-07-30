from __future__ import annotations

import json
import re
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parents[3] / "research" / "pseudo_xg" / "result.json"


def _result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_historical_result_declares_source_design_and_four_roles() -> None:
    result = _result()

    assert result["schema_version"] == "1.0"
    assert result["experiment_id"] == "pseudo_xg_lr64_historical_reproduction"
    assert result["status"] == "reproduced"
    assert result["class_order"] == ["H", "D", "A"]
    assert result["pseudo_xg"]["feature_count"] == 12  # type: ignore[index]
    assert [fold["role"] for fold in result["folds"]] == [  # type: ignore[index]
        "development",
        "development",
        "development",
        "historical_final",
    ]


def test_result_records_exact_failed_gate_and_source_disposition() -> None:
    result = _result()
    gate = result["success_gate"]

    assert gate["definition"]["minimum_improvement_log_loss"] == 0.005  # type: ignore[index]
    assert gate["definition"]["minimum_improving_folds"] == 2  # type: ignore[index]
    assert gate["observed"]["improving_fold_count"] == 1  # type: ignore[index]
    assert not gate["passed"]  # type: ignore[index]
    assert result["verdict"] == "failed_gate"
    assert result["disposition"] == "XG_SIGNAL_REJECTED_FOR_NOW"


def test_reference_comparison_is_explicit_and_within_tolerance() -> None:
    comparison = _result()["reference_comparison"]

    assert comparison["reproduction_passed"]  # type: ignore[index]
    tolerances = comparison["comparison_tolerances"]  # type: ignore[index]
    for differences in comparison["absolute_differences"]["folds"].values():  # type: ignore[index,union-attr]
        assert differences["log_loss"] <= tolerances["log_loss"]
        assert differences["brier_score"] <= tolerances["brier_score"]
        assert differences["ece"] <= tolerances["ece"]


def test_result_contains_no_paths_teams_predictions_coefficients_or_market_fields() -> None:
    text = RESULT_PATH.read_text(encoding="utf-8")
    result = _result()

    assert not re.search(r"[A-Za-z]:\\\\|/home/|/Users/", text)
    assert "HomeTeam" not in text
    assert "AwayTeam" not in text
    forbidden_keys = {
        "bankroll",
        "bets",
        "coefficients",
        "odds",
        "predictions",
        "provider",
        "raw_rows",
        "roi",
        "staking",
        "team_names",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert forbidden_keys.isdisjoint(keys(result))
    assert text.endswith("\n")
