from __future__ import annotations

import pandas as pd
import pytest

from fbai.data.schema import (
    CANONICAL_COLUMNS,
    NATURAL_KEY,
    CanonicalSchemaError,
    normalize_season_start_year,
    validate_canonical_frame,
)
from fbai.testing.synthetic import make_synthetic_canonical_matches


def canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=8,
        season_start_years=(2023,),
        divisions=("SYN1",),
        teams_per_division=4,
    )


def test_canonical_schema_uses_authoritative_stat_names() -> None:
    assert CANONICAL_COLUMNS == (
        "MatchDate",
        "SeasonStartYear",
        "Division",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "FTHome",
        "FTAway",
        "HomeShots",
        "AwayShots",
        "HomeTarget",
        "AwayTarget",
        "HomeCorners",
        "AwayCorners",
        "HomeFouls",
        "AwayFouls",
        "HomeYellow",
        "AwayYellow",
        "HomeRed",
        "AwayRed",
    )
    assert NATURAL_KEY == ("MatchDate", "Division", "HomeTeam", "AwayTeam")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2023, 2023), ("2023", 2023), ("2023-2024", 2023), ("2023/2024", 2023)],
)
def test_season_values_normalize(value: object, expected: int) -> None:
    assert normalize_season_start_year(value) == expected


@pytest.mark.parametrize("value", ["2324", "2023-2025", "23/24", "", None])
def test_malformed_or_ambiguous_seasons_raise(value: object) -> None:
    with pytest.raises(CanonicalSchemaError, match="Malformed or ambiguous"):
        normalize_season_start_year(value)


def test_invalid_ftr_labels_raise() -> None:
    frame = canonical()
    frame.loc[0, "FTR"] = "W"

    with pytest.raises(CanonicalSchemaError, match="outside H/D/A"):
        validate_canonical_frame(frame)


def test_null_natural_key_raises() -> None:
    frame = canonical()
    frame.loc[0, "HomeTeam"] = pd.NA

    with pytest.raises(CanonicalSchemaError, match="non-empty strings"):
        validate_canonical_frame(frame)


def test_home_team_equal_to_away_team_raises() -> None:
    frame = canonical()
    frame.loc[0, "AwayTeam"] = frame.loc[0, "HomeTeam"]

    with pytest.raises(CanonicalSchemaError, match="must differ"):
        validate_canonical_frame(frame)


def test_duplicate_natural_key_raises() -> None:
    frame = canonical()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(CanonicalSchemaError, match="not unique"):
        validate_canonical_frame(duplicated)
