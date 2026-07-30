from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RESULT_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "closing_market_benchmark" / "result.json"
)


def load_result() -> dict[str, Any]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {
            nested for item in value.values() for nested in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _all_keys(item)}
    return set()


def test_market_reproduction_record_has_declared_contract() -> None:
    result = load_result()

    assert result["schema_version"] == "1.0"
    assert result["experiment_id"] == "closing_market_benchmark_historical_reproduction"
    assert result["benchmark_name"] == "closing market benchmark"
    assert result["status"] == "reproduced"
    assert result["class_order"] == ["H", "D", "A"]
    assert result["source_odds_columns"] == ["AvgCH", "AvgCD", "AvgCA"]
    assert result["market_timing"]["classification"] == "closing_near_kickoff"
    assert [fold["role"] for fold in result["folds"]] == [
        "development",
        "development",
        "development",
        "historical_final",
    ]


def test_coverage_and_identical_aligned_samples_are_recorded() -> None:
    result = load_result()
    coverage = result["coverage"]

    assert coverage["candidate_match_rows"] == 22617
    assert coverage["valid_market_rows"] == 22617
    assert coverage["aligned_rows"] == 22617
    assert coverage["duplicate_market_keys"] == 0
    assert coverage["incomplete_odds_rows"] == 0
    assert coverage["invalid_odds_rows"] == 0
    assert coverage["unmatched_match_keys"] == 0
    assert coverage["unmatched_market_keys"] == 0
    for fold in result["folds"]:
        assert fold["aligned_lr52"]["n_samples"] == fold["market"]["n_samples"]
        assert fold["aligned_rows"] == fold["market"]["n_samples"]
        assert fold["candidate_test_rows"] == fold["aligned_rows"] + fold["unaligned_rows"]


def test_reference_tolerance_differences_and_verdict_are_explicit() -> None:
    comparison = load_result()["reference_comparison"]

    assert comparison["authoritative_reference"] == ("FBAI_NEW/results/odds_movement_eval.json")
    assert comparison["comparison_tolerances"]["log_loss"] == 1e-12
    assert comparison["verdict"] == "pass"
    assert comparison["match_level_probability_parity"] == "unavailable"
    for differences in comparison["absolute_differences"]["folds"].values():
        assert differences["log_loss"] <= comparison["comparison_tolerances"]["log_loss"]
        assert differences["brier_score"] <= comparison["comparison_tolerances"]["brier_score"]
        assert differences["ece"] <= comparison["comparison_tolerances"]["ece"]


def test_record_contains_no_private_paths_teams_rows_predictions_or_betting_fields() -> None:
    text = RESULT_PATH.read_text(encoding="utf-8")
    result = load_result()
    keys = {key.lower() for key in _all_keys(result)}
    forbidden_keys = {
        "bankroll",
        "bet_count",
        "edge",
        "match_predictions",
        "odds_rows",
        "profit",
        "roi",
        "staking",
    }

    assert not re.search(r"[A-Za-z]:\\\\|/home/|/Users/", text)
    assert "Synthetic Club" not in text
    assert forbidden_keys.isdisjoint(keys)
    assert text.endswith("\n")
