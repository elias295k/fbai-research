from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fbai.evaluation.market as market_module
from fbai.core.metrics import CLASS_ORDER
from fbai.evaluation.market import (
    AUTHORITATIVE_SOURCE_ODDS_COLUMNS,
    CANONICAL_MARKET_COLUMNS,
    CANONICAL_ODDS_COLUMNS,
    MARKET_CLASS_ORDER,
    MARKET_OVERROUND_COLUMN,
    MARKET_PROBABILITY_COLUMNS,
    MARKET_PROBABILITY_OUTPUT_COLUMNS,
    MarketSchemaError,
    closing_market_probabilities,
    normalize_closing_market,
    prepare_closing_market,
)
from fbai.testing.market import make_synthetic_closing_market
from fbai.testing.synthetic import make_synthetic_canonical_matches


def source_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["SYN2", "SYN1"],
            "Date": ["02/08/2023", "01/08/2023"],
            "HomeTeam": ["Synthetic Club 03", "Synthetic Club 01"],
            "AwayTeam": ["Synthetic Club 04", "Synthetic Club 02"],
            "AvgCH": [2.5, 2.0],
            "AvgCD": [3.2, 4.0],
            "AvgCA": [3.1, 5.0],
            "MaxCH": [2.6, 2.1],
            "FTR": ["A", "H"],
            "provider_note": ["ignored", "ignored"],
        }
    )


def test_authoritative_aliases_normalize_to_exact_stable_contract() -> None:
    market = source_market()

    normalized = normalize_closing_market(market)

    assert AUTHORITATIVE_SOURCE_ODDS_COLUMNS == ("AvgCH", "AvgCD", "AvgCA")
    assert tuple(normalized.columns) == CANONICAL_MARKET_COLUMNS
    assert tuple(normalized.loc[:, list(CANONICAL_ODDS_COLUMNS)].iloc[0]) == (2.0, 4.0, 5.0)
    assert normalized["MatchDate"].dtype == "datetime64[ns]"
    assert normalized["Division"].tolist() == ["SYN1", "SYN2"]
    assert "FTR" not in normalized
    assert "MaxCH" not in normalized
    assert "provider_note" not in normalized


def test_normalization_does_not_mutate_input() -> None:
    market = source_market()
    before = market.copy(deep=True)

    normalize_closing_market(market)

    pd.testing.assert_frame_equal(market, before)


def test_numeric_strings_normalize_to_float_odds() -> None:
    market = source_market()
    for column in ("AvgCH", "AvgCD", "AvgCA"):
        market[column] = market[column].astype(str)

    normalized = normalize_closing_market(market)

    assert all(pd.api.types.is_float_dtype(normalized[column]) for column in CANONICAL_ODDS_COLUMNS)


def test_missing_required_market_field_raises_precisely() -> None:
    with pytest.raises(MarketSchemaError, match="ClosingDrawOdds"):
        normalize_closing_market(source_market().drop(columns="AvgCD"))


@pytest.mark.parametrize("column", ["Date", "Div", "HomeTeam", "AwayTeam"])
def test_null_natural_keys_raise(column: str) -> None:
    market = source_market()
    market.loc[0, column] = None

    with pytest.raises(MarketSchemaError):
        normalize_closing_market(market)


def test_duplicate_natural_keys_raise_in_strict_and_coverage_paths() -> None:
    market = pd.concat([source_market(), source_market().iloc[[0]]], ignore_index=True)

    with pytest.raises(MarketSchemaError, match="not unique"):
        normalize_closing_market(market)
    with pytest.raises(MarketSchemaError, match="not unique"):
        prepare_closing_market(market)


@pytest.mark.parametrize("column", ["AvgCH", "AvgCD", "AvgCA"])
@pytest.mark.parametrize("value", [1.0, 0.0, -3.0])
def test_odds_not_strictly_greater_than_one_raise(column: str, value: float) -> None:
    market = source_market()
    market.loc[0, column] = value

    with pytest.raises(MarketSchemaError, match="strictly greater"):
        normalize_closing_market(market)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nan_and_infinite_odds_raise(value: float) -> None:
    market = source_market()
    market.loc[0, "AvgCH"] = value

    with pytest.raises(MarketSchemaError):
        normalize_closing_market(market)


def test_explicit_preparation_counts_incomplete_and_invalid_rows() -> None:
    market = source_market()
    extra = pd.concat([market, market.copy()], ignore_index=True)
    extra.loc[2, ["Date", "HomeTeam"]] = ["03/08/2023", "Synthetic Club 05"]
    extra.loc[2, "AvgCD"] = np.nan
    extra.loc[3, ["Date", "HomeTeam"]] = ["04/08/2023", "Synthetic Club 06"]
    extra.loc[3, "AvgCA"] = 1.0

    valid, audit = prepare_closing_market(extra)

    assert len(valid) == 2
    assert audit.supplied_market_rows == 4
    assert audit.valid_market_rows == 2
    assert audit.incomplete_odds_rows == 1
    assert audit.invalid_odds_rows == 1
    assert audit.duplicate_market_keys == 0


def test_de_vig_arithmetic_overround_and_named_hda_order_are_exact() -> None:
    one = source_market().iloc[[1]].copy()
    probabilities = closing_market_probabilities(one)
    raw = np.array([1.0 / 2.0, 1.0 / 4.0, 1.0 / 5.0])
    expected = raw / raw.sum()

    assert MARKET_CLASS_ORDER == CLASS_ORDER == ("H", "D", "A")
    assert tuple(probabilities.columns) == MARKET_PROBABILITY_OUTPUT_COLUMNS
    assert tuple(probabilities.loc[0, list(MARKET_PROBABILITY_COLUMNS)]) == pytest.approx(
        expected,
        abs=1e-15,
    )
    assert probabilities.loc[0, MARKET_OVERROUND_COLUMN] == pytest.approx(raw.sum(), abs=1e-15)


def test_probabilities_are_finite_bounded_normalized_and_allow_underround() -> None:
    market = source_market()
    market.loc[0, ["AvgCH", "AvgCD", "AvgCA"]] = [4.0, 4.0, 4.0]

    probabilities = closing_market_probabilities(market)
    values = probabilities.loc[:, list(MARKET_PROBABILITY_COLUMNS)].to_numpy()

    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0)).all()
    np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    assert probabilities.loc[1, MARKET_OVERROUND_COLUMN] == pytest.approx(0.75)


def test_shuffled_input_produces_identical_keyed_probabilities() -> None:
    expected = closing_market_probabilities(source_market())
    shuffled = closing_market_probabilities(
        source_market().sample(frac=1.0, random_state=9).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(shuffled, expected)


def test_invalid_implied_sum_guard_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        market_module.np,
        "reciprocal",
        lambda values: np.zeros_like(values),
    )

    with pytest.raises(MarketSchemaError, match="strictly positive"):
        closing_market_probabilities(source_market())


def test_synthetic_market_is_separate_valid_and_seed_reproducible() -> None:
    matches = make_synthetic_canonical_matches(
        seed=91,
        season_start_years=(2023,),
        divisions=("SYN1",),
        teams_per_division=4,
    )

    first = make_synthetic_closing_market(matches, seed=92)
    second = make_synthetic_closing_market(matches, seed=92)
    different = make_synthetic_closing_market(matches, seed=93)

    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(different)
    assert tuple(first.columns) == CANONICAL_MARKET_COLUMNS
    assert (first.loc[:, list(CANONICAL_ODDS_COLUMNS)] > 1.0).all().all()
    assert not set(CANONICAL_ODDS_COLUMNS).intersection(matches.columns)
    normalize_closing_market(first)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing_rows", "fewer"),
        ("incomplete", "incomplete"),
        ("duplicate", "duplicate"),
        ("invalid", "invalid"),
        ("shuffled", "valid"),
    ],
)
def test_synthetic_market_negative_and_shuffle_modes_are_explicit(
    mode: str,
    expected: str,
) -> None:
    matches = make_synthetic_canonical_matches(
        seed=94,
        season_start_years=(2023,),
        divisions=("SYN1",),
        teams_per_division=4,
    )
    market = make_synthetic_closing_market(matches, seed=95, mode=mode)  # type: ignore[arg-type]

    if expected == "fewer":
        assert len(market) < len(matches)
    elif expected == "incomplete":
        _, audit = prepare_closing_market(market)
        assert audit.incomplete_odds_rows == 1
    elif expected == "duplicate":
        with pytest.raises(MarketSchemaError, match="not unique"):
            prepare_closing_market(market)
    elif expected == "invalid":
        _, audit = prepare_closing_market(market)
        assert audit.invalid_odds_rows == 1
    else:
        normalize_closing_market(market)
