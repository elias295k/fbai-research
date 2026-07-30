from __future__ import annotations

from pathlib import Path

import pytest

from fbai.data.sources import (
    UnsupportedDivisionError,
    UnsupportedSeasonError,
    build_source_url,
    season_label,
    season_start_year,
    source_csv_path,
)


def test_supported_source_url_is_deterministic() -> None:
    assert build_source_url("2324", "E0") == ("https://www.football-data.co.uk/mmz4281/2324/E0.csv")
    assert build_source_url("2324", "E0") == build_source_url("2324", "E0")


def test_unsupported_division_raises() -> None:
    with pytest.raises(UnsupportedDivisionError, match="Unsupported division"):
        build_source_url("2324", "SYN1")


def test_unsupported_season_raises() -> None:
    with pytest.raises(UnsupportedSeasonError, match="Unsupported season"):
        build_source_url("9999", "E0")


def test_source_season_metadata_is_explicit() -> None:
    assert season_start_year("2021") == 2020
    assert season_label("2021") == "2020-2021"


def test_local_source_path_is_deterministic(tmp_path: Path) -> None:
    assert source_csv_path(tmp_path, "2324", "E0") == tmp_path / "2324" / "E0.csv"
