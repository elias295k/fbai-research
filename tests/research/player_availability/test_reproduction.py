from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fbai.features import FEATURE_COLUMNS
from fbai.research.player_availability.schema import PRIOR_AVAILABILITY_FEATURES

ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = ROOT / "research" / "player_availability" / "result.json"


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_reproduction_record_has_exact_authority_scope_and_features(result) -> None:
    assert result["status"] == "reproduced"
    assert result["source_authority"]["executable"] == [
        "src/fbai_new/availability.py",
        "scripts/run_availability_eval.py",
    ]
    assert result["scope"]["base_big5_matches"] == 12458
    assert result["scope"]["covered_matches"] == 12452
    assert result["scope"]["coverage_filter_changes_lr52_population"]
    assert [result["folds"][str(year)]["test_rows"] for year in range(2022, 2026)] == [
        1825,
        1751,
        1752,
        1751,
    ]
    assert tuple(result["feature_contract"]["features"]) == PRIOR_AVAILABILITY_FEATURES
    assert result["feature_contract"]["candidate_input_count"] == 115


def test_timing_audit_excludes_unproven_target_lineup(result) -> None:
    timing = result["timing_audit"]

    assert timing["target_official_lineup_and_bench"] == "timing_unknown_forbidden"
    assert timing["target_participation_minutes_and_substitutions"] == (
        "known_only_after_target_match_forbidden"
    )
    assert timing["injury_suspension_absence_labels"] == "timing_unknown_not_available"
    assert timing["same_date_rows"] == ("forbidden_until_entire_date_batch_is_transformed")


def test_reference_metrics_reproduce_within_declared_tolerance(result) -> None:
    source = {
        "2022": (1.017802397311472, 0.6046228929865521, 0.034346320775909614),
        "2023": (0.9804204145445805, 0.5832130566377514, 0.024041582716821303),
        "2024": (0.9989451743448836, 0.596220001370218, 0.024000719768375887),
        "2025": (1.0010733169623836, 0.5960671566112994, 0.020477559666628137),
    }
    tolerance = result["reference_comparison"]["comparison_tolerances"]
    for year, expected in source.items():
        observed = result["folds"][year]["candidate"]
        assert observed["log_loss"] == pytest.approx(expected[0], abs=tolerance["log_loss"])
        assert observed["brier_score"] == pytest.approx(expected[1], abs=tolerance["brier_score"])
        assert observed["ece"] == pytest.approx(expected[2], abs=tolerance["ece"])
    assert result["reference_comparison"]["safe_feature_cells_changed_vs_source_cache"] == 0
    assert result["reference_comparison"]["reproduction_passed"]


def test_gate_disposition_privacy_and_stable_contract(result) -> None:
    gate = result["success_gate"]
    expected = (
        result["development"]["lr52"]["log_loss"] - result["development"]["candidate"]["log_loss"]
    )

    assert gate["observed"]["candidate_improvement_log_loss"] == pytest.approx(expected)
    assert gate["observed"]["improving_fold_count"] == 1
    assert not gate["passed"]
    assert result["disposition"] == "PLAYER_AVAILABILITY_REJECTED_FOR_NOW"
    assert not result["retention"]["individual_player_rows_exported"]
    assert not result["retention"]["predictions_exported"]
    serialized = json.dumps(result).lower()
    assert "c:\\" not in serialized
    assert ".csv" not in serialized
    assert ".parquet" not in serialized
    fingerprint = hashlib.sha256("\0".join(FEATURE_COLUMNS).encode()).hexdigest()
    assert fingerprint == result["feature_contract"]["stable_feature_fingerprint"]
