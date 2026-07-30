from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fbai.data.canonical import write_canonical_partitions
from fbai.data.schema import CANONICAL_COLUMNS, CanonicalSchemaError
from fbai.testing.synthetic import make_synthetic_canonical_matches


def canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=18,
        season_start_years=(2022, 2023),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )


def test_one_verified_parquet_file_is_written_per_division(tmp_path: Path) -> None:
    frame = canonical()

    result = write_canonical_partitions(frame, tmp_path)

    assert [partition.division for partition in result.partitions] == ["SYN1", "SYN2"]
    assert [partition.path.name for partition in result.partitions] == [
        "SYN1.parquet",
        "SYN2.parquet",
    ]
    assert result.input_row_count == len(frame)
    assert result.output_row_count == len(frame)
    assert sum(partition.row_count for partition in result.partitions) == len(frame)


def test_parquet_round_trip_preserves_schema_rows_and_order(tmp_path: Path) -> None:
    frame = canonical()
    result = write_canonical_partitions(frame, tmp_path)

    round_trip = pd.concat(
        [pd.read_parquet(partition.path) for partition in result.partitions],
        ignore_index=True,
    )
    expected = (
        frame.sort_values(
            ["Division", "MatchDate", "HomeTeam", "AwayTeam"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .loc[:, list(CANONICAL_COLUMNS)]
    )
    actual = (
        round_trip.sort_values(
            ["Division", "MatchDate", "HomeTeam", "AwayTeam"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .loc[:, list(CANONICAL_COLUMNS)]
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_shuffled_input_produces_identical_partition_rows(tmp_path: Path) -> None:
    frame = canonical()
    first = write_canonical_partitions(frame, tmp_path / "first")
    second = write_canonical_partitions(
        frame.sample(frac=1.0, random_state=77),
        tmp_path / "second",
    )

    for first_partition, second_partition in zip(
        first.partitions,
        second.partitions,
        strict=True,
    ):
        pd.testing.assert_frame_equal(
            pd.read_parquet(first_partition.path),
            pd.read_parquet(second_partition.path),
        )
        assert first_partition.path.read_bytes() == second_partition.path.read_bytes()


def test_failed_validation_produces_no_final_artifact(tmp_path: Path) -> None:
    invalid = canonical().drop(columns="HomeShots")
    destination = tmp_path / "canonical"

    with pytest.raises(CanonicalSchemaError, match="Missing canonical"):
        write_canonical_partitions(invalid, destination)

    assert not destination.exists()
