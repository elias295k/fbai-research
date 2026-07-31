from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fbai.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = ROOT / "research" / "graph_model" / "result.json"


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_reproduction_record_has_exact_authority_and_configuration(result) -> None:
    assert result["status"] == "reproduced"
    assert result["stable_feature_count"] == 52
    assert result["scope"]["matched_graph_rows"] == 12452
    assert result["scope"]["fold_test_rows"] == {
        "2022": 1825,
        "2023": 1751,
        "2024": 1752,
        "2025": 1751,
    }
    assert [
        (config["name"], config["embedding_dimension"], config["graph_feature_count"])
        for config in result["configurations"]
    ] == [
        ("svd8_recent10", 8, 40),
        ("svd16_recent10", 16, 72),
    ]
    assert result["selection"]["selected_configuration"] == "svd8_recent10"
    assert result["selection"]["selected_context"] == "lr52_graph"


def test_selected_internal_candidate_reproduces_source_within_tolerance(result) -> None:
    selected = next(
        candidate
        for candidate in result["development_candidates"]
        if candidate["configuration"] == "svd8_recent10" and candidate["context"] == "lr52_graph"
    )
    source_folds = {
        "2022": (1.0069674903438277, 0.6010952386195972, 0.023738111730367894),
        "2023": (0.9828525638715074, 0.5854293374584367, 0.020503049727217986),
        "2024": (0.9915637059156837, 0.5913510757583439, 0.020371910040384526),
    }
    tolerance = result["reference_comparison"]["comparison_tolerances"]
    for year, reference in source_folds.items():
        observed = selected["folds"][year]
        assert observed["log_loss"] == pytest.approx(reference[0], abs=tolerance["log_loss"])
        assert observed["brier_score"] == pytest.approx(reference[1], abs=tolerance["brier_score"])
        assert observed["ece"] == pytest.approx(reference[2], abs=tolerance["ece"])
    final = result["historical_final_contexts"]["lr52_graph"]
    assert final["log_loss"] == pytest.approx(1.0005311955056302, abs=1e-8)
    assert result["reference_comparison"]["reproduction_passed"]


def test_gate_math_and_disposition_are_source_exact(result) -> None:
    gate = result["success_gate"]
    selected = next(
        candidate
        for candidate in result["development_candidates"]
        if candidate["configuration"] == "svd8_recent10" and candidate["context"] == "lr52_graph"
    )
    expected = (
        result["baseline_lr52"]["development"]["log_loss"] - selected["development"]["log_loss"]
    )

    assert gate["definition"]["minimum_improvement_log_loss"] == 0.005
    assert gate["definition"]["minimum_improving_folds"] == 2
    assert gate["observed"]["candidate_improvement_log_loss"] == pytest.approx(expected)
    assert gate["observed"]["improving_fold_count"] == 2
    assert not gate["passed"]
    assert not gate["final_confirmed"]
    assert result["disposition"] == "DATA_CEILING_UPHELD"


def test_result_is_aggregate_only_and_stable_contract_is_unchanged(result) -> None:
    serialized = json.dumps(result).lower()
    forbidden_keys = {
        "absolute_path",
        "paths",
        "team_names",
        "node_names",
        "edge_lists",
        "predictions",
        "weights",
        "tensors",
        "odds",
        "roi",
        "betting",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(key.lower() for key in value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(result)
    assert "c:\\" not in serialized
    assert ".csv" not in serialized
    assert ".parquet" not in serialized
    fingerprint = hashlib.sha256("\0".join(FEATURE_COLUMNS).encode()).hexdigest()
    assert fingerprint == ("09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78")
