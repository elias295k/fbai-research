from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from fbai.data.canonical import write_canonical_partitions
from fbai.data.store import (
    canonical_connection,
    discover_canonical_files,
    query_canonical,
)
from fbai.testing.synthetic import make_synthetic_canonical_matches


def write_partitions(directory: Path) -> int:
    frame = make_synthetic_canonical_matches(
        seed=35,
        season_start_years=(2023,),
        divisions=("SYN1", "SYN2"),
        teams_per_division=4,
    )
    write_canonical_partitions(frame, directory)
    return len(frame)


def test_canonical_files_are_queryable(tmp_path: Path) -> None:
    row_count = write_partitions(tmp_path)

    result = query_canonical(
        tmp_path,
        "SELECT COUNT(*) AS n FROM matches",
    )

    assert int(result.loc[0, "n"]) == row_count


def test_multiple_division_files_are_unioned_by_name(tmp_path: Path) -> None:
    write_partitions(tmp_path)

    result = query_canonical(
        tmp_path,
        "SELECT Division, COUNT(*) AS n FROM matches GROUP BY Division ORDER BY Division",
    )

    assert result["Division"].tolist() == ["SYN1", "SYN2"]
    assert (result["n"] > 0).all()
    assert [path.name for path in discover_canonical_files(tmp_path)] == [
        "SYN1.parquet",
        "SYN2.parquet",
    ]


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_canonical_files(tmp_path / "missing")


def test_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="contains no Parquet"):
        discover_canonical_files(empty)


def test_connection_is_closed_after_context_exit(tmp_path: Path) -> None:
    write_partitions(tmp_path)

    with canonical_connection(tmp_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone() is not None

    with pytest.raises(duckdb.Error):
        connection.execute("SELECT 1")
