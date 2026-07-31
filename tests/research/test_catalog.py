from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fbai.features import FEATURE_COLUMNS
from fbai.research.catalog import (
    PROGRAMME_CONCLUSION,
    RESULT_ALLOWLIST,
    CatalogValidationError,
    CrossPopulationComparisonError,
    ExperimentRole,
    build_research_catalog,
    load_research_catalog,
    rank_same_population_by_raw_log_loss,
)

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "research" / "summary" / "result.json"


def _records() -> dict[str, dict[str, Any]]:
    return {
        spec.record_id: json.loads((ROOT / spec.relative_file).read_text(encoding="utf-8"))
        for spec in RESULT_ALLOWLIST
    }


@pytest.fixture(scope="module")
def catalog():
    return load_research_catalog(ROOT)


def test_explicit_allowlist_contains_every_expected_record(catalog) -> None:
    assert [spec.record_id for spec in RESULT_ALLOWLIST] == [
        "lr52_baseline",
        "closing_market_benchmark",
        "match2vec",
        "pseudo_xg",
        "understat_xg",
        "deep_capacity",
        "graph_model",
        "player_availability",
    ]
    assert len(catalog.candidates) == 6
    assert len(catalog.protected_record_hashes) == 8


def test_duplicate_experiment_ids_and_missing_required_fields_fail() -> None:
    duplicate = copy.deepcopy(_records())
    duplicate["pseudo_xg"]["experiment_id"] = duplicate["match2vec"]["experiment_id"]
    with pytest.raises(CatalogValidationError, match="duplicate experiment IDs"):
        build_research_catalog(duplicate)

    missing = copy.deepcopy(_records())
    del missing["match2vec"]["development"]
    with pytest.raises(CatalogValidationError, match="missing required field development"):
        build_research_catalog(missing)


def test_improvements_are_recomputed_with_the_correct_sign(catalog) -> None:
    expected = {
        "Match2Vec": 0.0037639960892689173,
        "Graph SVD8": 0.0014562398332772508,
        "Deep MLP": 0.00016412386273456647,
        "Pseudo-xG": 0.000038973735167036061,
        "Player availability": -0.003893605177992421,
        "Understat xG": -0.004516960757590671,
    }
    observed = {candidate.name: candidate for candidate in catalog.candidates}

    for name, improvement in expected.items():
        candidate = observed[name]
        assert candidate.candidate_improvement_log_loss == pytest.approx(improvement)
        assert candidate.candidate_improvement_log_loss == pytest.approx(
            candidate.aligned_lr52_log_loss - candidate.candidate_log_loss
        )


def test_gate_verdicts_match_thresholds_and_fold_counts(catalog) -> None:
    for candidate in catalog.candidates:
        expected = (
            candidate.candidate_improvement_log_loss
            >= candidate.gate_policy.minimum_improvement_log_loss
            and candidate.improving_development_folds
            >= candidate.gate_policy.minimum_improving_folds
        )
        assert candidate.gate_passed is expected
        assert not candidate.gate_passed
        assert not candidate.selected_as_default


def test_closing_market_is_external_and_lr52_remains_default(catalog) -> None:
    assert catalog.external_benchmark.role is ExperimentRole.EXTERNAL_BENCHMARK
    assert not catalog.external_benchmark.gate_applicable
    assert catalog.baseline.role is ExperimentRole.BASELINE
    assert catalog.baseline.selected_as_default
    assert catalog.default_model == "LR52"
    assert catalog.conclusion == PROGRAMME_CONCLUSION


def test_different_populations_cannot_be_ranked_by_raw_log_loss(catalog) -> None:
    with pytest.raises(CrossPopulationComparisonError):
        catalog.rank_candidates_by_raw_log_loss()
    with pytest.raises(CrossPopulationComparisonError):
        rank_same_population_by_raw_log_loss(catalog.candidates)


def test_valid_ranking_uses_only_within_experiment_improvement(catalog) -> None:
    ranked_names = {candidate.experiment_id: candidate.name for candidate in catalog.candidates}
    assert [
        ranked_names[experiment_id]
        for experiment_id in catalog.candidate_ranking_by_valid_development_improvement
    ] == [
        "Match2Vec",
        "Graph SVD8",
        "Deep MLP",
        "Pseudo-xG",
        "Player availability",
        "Understat xG",
    ]


def test_generated_summary_matches_catalog_and_contains_no_sensitive_artifacts(catalog) -> None:
    committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert committed == catalog.to_summary_dict()
    serialized = json.dumps(committed).lower()
    forbidden_keys = {
        "path",
        "paths",
        "team_names",
        "player_names",
        "raw_data",
        "odds_rows",
        "predictions",
        "roi",
        "betting",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(str(key).lower() for key in value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(committed)
    assert "c:\\" not in serialized
    assert ".csv" not in serialized
    assert ".parquet" not in serialized


def test_protected_records_are_byte_exact(catalog) -> None:
    expected = {item.record_id: item.sha256 for item in catalog.protected_record_hashes}
    actual = {
        spec.record_id: hashlib.sha256((ROOT / spec.relative_file).read_bytes()).hexdigest()
        for spec in RESULT_ALLOWLIST
    }
    committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    recorded = {item["record_id"]: item["sha256"] for item in committed["protected_record_hashes"]}

    assert expected == actual == recorded


def test_stable_feature_contract_and_fingerprint_are_unchanged() -> None:
    assert len(FEATURE_COLUMNS) == 52
    fingerprint = hashlib.sha256("\0".join(FEATURE_COLUMNS).encode()).hexdigest()
    assert fingerprint == "09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78"
