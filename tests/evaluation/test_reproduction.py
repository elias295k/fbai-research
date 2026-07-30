from __future__ import annotations

import json
import re
from pathlib import Path

RESULT_PATH = Path(__file__).resolve().parents[2] / "research" / "lr52_baseline" / "result.json"


def load_result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_reproduction_record_has_declared_contract() -> None:
    result = load_result()

    assert result["schema_version"] == "1.0"
    assert result["experiment_id"] == "lr52_baseline_historical_reproduction"
    assert result["model_name"] == "LR52"
    assert result["feature_count"] == 52
    assert result["class_order"] == ["H", "D", "A"]
    assert len(result["folds"]) == 4  # type: ignore[arg-type]
    assert [fold["role"] for fold in result["folds"]] == [  # type: ignore[index]
        "development",
        "development",
        "development",
        "historical_final",
    ]


def test_reference_comparison_declares_tolerance_and_verdict() -> None:
    comparison = load_result()["reference_comparison"]

    assert comparison["reference_weighted_log_loss"] == 0.9953674687082226  # type: ignore[index]
    assert comparison["absolute_difference"] >= 0.0  # type: ignore[index]
    assert comparison["tolerance"] > 0.0  # type: ignore[index]
    assert comparison["verdict"] in {"pass", "fail"}  # type: ignore[index]
    assert comparison["match_level_prediction_parity"] == "unavailable"  # type: ignore[index]


def test_record_contains_no_paths_raw_rows_or_team_names() -> None:
    text = RESULT_PATH.read_text(encoding="utf-8")
    result = load_result()

    assert not re.search(r"[A-Za-z]:\\\\|/home/|/Users/", text)
    assert "Synthetic Club" not in text
    assert "HomeTeam" not in text
    assert "AwayTeam" not in text
    assert "predictions" not in result
    assert "raw_rows" not in result
    assert text.endswith("\n")
