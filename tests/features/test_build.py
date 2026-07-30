from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import fbai.features.checks as feature_checks
from fbai.data.canonical import write_canonical_partitions
from fbai.features.build import (
    build_feature_table,
    build_feature_table_from_parquet,
    write_feature_partitions,
)
from fbai.features.checks import FeatureValidationError, validate_feature_table
from fbai.features.labels import TARGET_COLUMNS
from fbai.features.schema import FEATURE_COLUMNS, FEATURE_TABLE_COLUMNS, METADATA_COLUMNS
from fbai.testing.synthetic import make_synthetic_canonical_matches

KEY = ["MatchDate", "Division", "HomeTeam", "AwayTeam"]


def canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=202,
        season_start_years=(2022, 2023),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(KEY).reset_index(drop=True)


def test_build_preserves_rows_contract_and_source_frame() -> None:
    source = canonical()
    before = source.copy(deep=True)

    result = build_feature_table(source)

    assert len(result) == len(source)
    assert tuple(result.columns) == FEATURE_TABLE_COLUMNS
    assert len(FEATURE_COLUMNS) == 52
    assert set(METADATA_COLUMNS).issubset(result.columns)
    assert set(TARGET_COLUMNS).issubset(result.columns)
    validate_feature_table(result, expected_row_count=len(source))
    pd.testing.assert_frame_equal(source, before)


def test_one_input_row_produces_one_feature_row() -> None:
    source = canonical().iloc[[0]].reset_index(drop=True)

    result = build_feature_table(source)

    assert len(result) == 1
    assert result.loc[0, list(FEATURE_COLUMNS)].isna().sum() == 45


def test_shuffled_input_builds_identical_output() -> None:
    source = canonical()

    first = build_feature_table(source)
    second = build_feature_table(source.sample(frac=1.0, random_state=44))

    pd.testing.assert_frame_equal(first, second)


def test_build_from_canonical_parquet_directory(tmp_path: Path) -> None:
    source = canonical()
    canonical_directory = tmp_path / "canonical"
    write_canonical_partitions(source, canonical_directory)

    expected = build_feature_table(source)
    actual = build_feature_table_from_parquet(canonical_directory)

    pd.testing.assert_frame_equal(actual, expected)


def test_one_verified_feature_file_per_division_and_round_trip(tmp_path: Path) -> None:
    features = build_feature_table(canonical())

    result = write_feature_partitions(features, tmp_path / "features")

    assert [partition.division for partition in result.partitions] == ["SYN1", "SYN2"]
    assert [partition.path.name for partition in result.partitions] == [
        "SYN1.parquet",
        "SYN2.parquet",
    ]
    assert result.input_row_count == len(features)
    assert result.output_row_count == len(features)
    round_trip = pd.concat(
        [pd.read_parquet(partition.path) for partition in result.partitions],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(aligned(round_trip), aligned(features))


def test_failed_validation_leaves_no_destination(tmp_path: Path) -> None:
    invalid = build_feature_table(canonical()).drop(columns=FEATURE_COLUMNS[-1])
    destination = tmp_path / "features"

    with pytest.raises(FeatureValidationError):
        write_feature_partitions(invalid, destination)

    assert not destination.exists()


def test_public_build_and_write_paths_invoke_semantic_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    original = feature_checks.assert_model_inputs_safe

    def recording_guard(
        columns: tuple[str, ...],
        *,
        approved_pre_features: tuple[str, ...],
    ) -> None:
        calls.append(tuple(columns))
        original(columns, approved_pre_features=approved_pre_features)

    monkeypatch.setattr(feature_checks, "assert_model_inputs_safe", recording_guard)

    features = build_feature_table(canonical())
    build_calls = len(calls)
    write_feature_partitions(features, tmp_path / "features")

    assert build_calls >= 1
    assert len(calls) > build_calls
    assert all(call == FEATURE_COLUMNS for call in calls)
