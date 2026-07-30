"""Chronological evaluation splits with deterministic, same-date-safe ordering."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

STABLE_ORDER_KEY: tuple[str, ...] = ("MatchDate", "Division", "HomeTeam", "AwayTeam")
DEVELOPMENT_TEST_YEARS: tuple[int, ...] = (2022, 2023, 2024)
HISTORICAL_FINAL_TEST_YEAR = 2025
ALL_TEST_YEARS: tuple[int, ...] = (*DEVELOPMENT_TEST_YEARS, HISTORICAL_FINAL_TEST_YEAR)


class FoldRole(StrEnum):
    """Evaluation role assigned to each outer fold."""

    DEVELOPMENT = "development"
    HISTORICAL_FINAL = "historical_final"


@dataclass(frozen=True)
class ChronologicalFold:
    """Indices and metadata for one expanding-window evaluation fold."""

    test_year: int
    role: FoldRole
    train_idx: tuple[object, ...]
    test_idx: tuple[object, ...]

    @property
    def name(self) -> str:
        return f"train_through_{self.test_year - 1}_test_{self.test_year}"


def _prepare_matches(
    matches: pd.DataFrame,
    *,
    date_column: str,
    season_column: str | None,
) -> pd.DataFrame:
    required = set(STABLE_ORDER_KEY)
    required.remove("MatchDate")
    required.add(date_column)
    if season_column is not None:
        required.add(season_column)
    missing = sorted(required.difference(matches.columns))
    if missing:
        raise ValueError(f"Missing split columns: {', '.join(missing)}")
    if not matches.index.is_unique:
        raise ValueError("Match index must be unique so split indices are unambiguous")

    prepared = matches.copy()
    stable_key = [date_column, "Division", "HomeTeam", "AwayTeam"]
    null_columns = [column for column in stable_key if prepared[column].isna().any()]
    if null_columns:
        raise ValueError(f"Stable ordering key contains nulls in: {', '.join(null_columns)}")

    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
    if season_column is not None:
        prepared[season_column] = pd.to_numeric(prepared[season_column], errors="raise").astype(int)
    return prepared.sort_values(stable_key, kind="mergesort")


def sort_matches(
    matches: pd.DataFrame,
    *,
    date_column: str = "MatchDate",
) -> pd.DataFrame:
    """Return matches in the stable chronological order required by the protocol."""

    return _prepare_matches(matches, date_column=date_column, season_column=None)


def expanding_folds(
    matches: pd.DataFrame,
    *,
    test_years: Sequence[int] = ALL_TEST_YEARS,
    date_column: str = "MatchDate",
    season_column: str = "SeasonStartYear",
) -> Iterator[ChronologicalFold]:
    """Yield expanding folds and enforce strict train-before-test chronology."""

    prepared = _prepare_matches(
        matches,
        date_column=date_column,
        season_column=season_column,
    )
    for test_year in test_years:
        train = prepared.loc[prepared[season_column] <= test_year - 1]
        test = prepared.loc[prepared[season_column] == test_year]
        if train.empty or test.empty:
            raise ValueError(f"Fold {test_year} has an empty train or test partition")
        latest_train = train[date_column].max()
        earliest_test = test[date_column].min()
        if latest_train >= earliest_test:
            raise ValueError(
                f"Fold {test_year} violates chronology: latest train date "
                f"{latest_train} is not earlier than earliest test date {earliest_test}"
            )
        role = (
            FoldRole.HISTORICAL_FINAL
            if test_year == HISTORICAL_FINAL_TEST_YEAR
            else FoldRole.DEVELOPMENT
        )
        yield ChronologicalFold(
            test_year=test_year,
            role=role,
            train_idx=tuple(train.index),
            test_idx=tuple(test.index),
        )


def inner_time_split(
    training_matches: pd.DataFrame,
    *,
    validation_fraction: float = 0.1,
    date_column: str = "MatchDate",
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Split training data at a date boundary, never dividing a calendar date."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    prepared = _prepare_matches(
        training_matches,
        date_column=date_column,
        season_column=None,
    )
    unique_dates = prepared[date_column].drop_duplicates().reset_index(drop=True)
    if len(unique_dates) < 2:
        raise ValueError("Inner split requires matches on at least two distinct dates")

    validation_date_count = max(1, int(len(unique_dates) * validation_fraction))
    validation_date_count = min(validation_date_count, len(unique_dates) - 1)
    first_validation_date = unique_dates.iloc[-validation_date_count]
    fit = prepared.loc[prepared[date_column] < first_validation_date]
    validation = prepared.loc[prepared[date_column] >= first_validation_date]
    if fit.empty or validation.empty:
        raise ValueError("Inner split produced an empty fit or validation partition")
    if fit[date_column].max() >= validation[date_column].min():
        raise ValueError("Inner split does not have strict date separation")
    return tuple(fit.index), tuple(validation.index)


def chronological_date_batches(
    matches: pd.DataFrame,
    *,
    date_column: str = "MatchDate",
) -> tuple[tuple[object, ...], ...]:
    """Return indivisible date batches in stable order.

    Stateful consumers must predict every match in a batch before incorporating
    any result from that batch.
    """

    prepared = _prepare_matches(matches, date_column=date_column, season_column=None)
    return tuple(
        tuple(batch.index) for _, batch in prepared.groupby(date_column, sort=False, observed=True)
    )
