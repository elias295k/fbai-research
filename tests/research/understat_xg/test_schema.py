from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.research.understat_xg.schema import (
    TEAM_ALIASES,
    UnderstatXGSchemaError,
    audit_understat_xg_frame,
    normalize_team_name,
    validate_understat_xg_frame,
)


def _external() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Division": ["E0", "D1"],
            "MatchDate": ["2023-08-12", "2023-08-13"],
            "HomeTeam": ["Manchester United", "Bayer Leverkusen"],
            "AwayTeam": ["Wolverhampton Wanderers", "Borussia M'gladbach"],
            "home_xg": ["1.25", 2.1],
            "away_xg": [0.75, "1.0"],
        }
    )


def test_schema_normalizes_dates_values_and_source_verified_aliases() -> None:
    normalized = validate_understat_xg_frame(_external())

    assert tuple(normalized["HomeTeam"]) == ("man united", "leverkusen")
    assert tuple(normalized["AwayTeam"]) == ("wolves", "mgladbach")
    assert pd.api.types.is_datetime64_any_dtype(normalized["MatchDate"])
    assert normalized[["home_xg", "away_xg"]].dtypes.apply(pd.api.types.is_numeric_dtype).all()
    assert TEAM_ALIASES["real valladolid"] == "valladolid"
    assert normalize_team_name("Paris Saint-Germain") == "paris sg"


def test_audit_reports_duplicate_invalid_and_unsupported_rows_without_keys() -> None:
    invalid = pd.concat([_external(), _external().iloc[[0]]], ignore_index=True)
    invalid.loc[1, "Division"] = "SYN"
    invalid.loc[1, "home_xg"] = np.inf
    audit = audit_understat_xg_frame(invalid)

    assert audit.supplied_rows == 3
    assert audit.duplicate_keys == 1
    assert audit.duplicate_rows == 2
    assert audit.invalid_xg_rows == 1
    assert audit.unsupported_division_rows == 1
    assert not audit.passed
    assert "HomeTeam" not in str(audit.to_dict())


@pytest.mark.parametrize("value", [-0.01, np.inf, -np.inf, np.nan, "not-a-number"])
def test_nonfinite_or_negative_xg_is_rejected(value: object) -> None:
    frame = _external()
    frame.loc[0, "home_xg"] = value

    with pytest.raises(UnderstatXGSchemaError, match="invalid_xg_rows=1"):
        validate_understat_xg_frame(frame)


def test_schema_is_closed() -> None:
    frame = _external().assign(provider_match_id=[1, 2])

    with pytest.raises(UnderstatXGSchemaError, match="Unexpected external xG columns"):
        validate_understat_xg_frame(frame)
