from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fbai.data.loader import (
    MissingSourceColumnsError,
    canonicalize_source_frame,
    load_source_csv,
)
from fbai.data.schema import CANONICAL_COLUMNS, CanonicalSchemaError
from fbai.testing.synthetic import make_synthetic_raw_matches


def raw() -> pd.DataFrame:
    return make_synthetic_raw_matches(
        seed=11,
        season_start_years=(2023,),
        divisions=("SYN1",),
        teams_per_division=4,
    )


def test_source_aliases_normalize_to_exact_canonical_schema() -> None:
    canonical = canonicalize_source_frame(raw())

    assert tuple(canonical.columns) == CANONICAL_COLUMNS
    assert {"FTHome", "FTAway", "HomeTarget", "AwayTarget"} <= set(canonical)
    assert not {"FTHG", "FTAG", "HST", "AST"} & set(canonical)


def test_supported_date_formats_normalize() -> None:
    source = raw()
    dates = pd.to_datetime(source.loc[:2, "Date"], dayfirst=True)
    source.loc[0, "Date"] = dates.iloc[0].strftime("%Y-%m-%d")
    source.loc[1, "Date"] = dates.iloc[1].strftime("%d/%m/%Y")
    source.loc[2, "Date"] = dates.iloc[2].strftime("%d/%m/%y")

    canonical = canonicalize_source_frame(source)

    assert str(canonical["MatchDate"].dtype) == "datetime64[ns]"
    assert canonical["MatchDate"].eq(canonical["MatchDate"].dt.normalize()).all()


def test_explicit_source_season_context_is_supported() -> None:
    source = raw().drop(columns="Season")

    canonical = canonicalize_source_frame(source, source_season_code="2324")

    assert canonical["SeasonStartYear"].eq(2023).all()


def test_missing_required_source_columns_raise_with_names() -> None:
    source = raw().drop(columns=["HS", "AR"])

    with pytest.raises(MissingSourceColumnsError) as exc_info:
        canonicalize_source_frame(source)

    assert "HomeShots" in str(exc_info.value)
    assert "AwayRed" in str(exc_info.value)


def test_invalid_numeric_field_raises() -> None:
    source = raw()
    source["HS"] = source["HS"].astype(object)
    source.loc[0, "HS"] = "not-a-number"

    with pytest.raises(CanonicalSchemaError, match="HomeShots contains non-numeric"):
        canonicalize_source_frame(source)


def test_unparseable_date_raises() -> None:
    source = raw()
    source.loc[0, "Date"] = "not-a-date"

    with pytest.raises(CanonicalSchemaError, match="Unparseable MatchDate"):
        canonicalize_source_frame(source)


def test_extra_source_columns_do_not_enter_canonical_output() -> None:
    source = raw().assign(
        B365H=1.9,
        attendance=50_000,
        database_id=123,
    )

    canonical = canonicalize_source_frame(source)

    assert tuple(canonical.columns) == CANONICAL_COLUMNS
    assert not {"B365H", "attendance", "database_id"} & set(canonical)


def test_local_csv_loader_is_bom_aware(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    raw().to_csv(path, index=False, encoding="utf-8-sig")

    canonical = load_source_csv(path)

    assert len(canonical) == len(raw())
