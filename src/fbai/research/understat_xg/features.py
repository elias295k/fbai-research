"""Strictly-prior historical Understat xG feature construction."""

from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fbai.core.leakage import assert_model_inputs_safe, assert_natural_key_valid
from fbai.data.schema import NATURAL_KEY, STABLE_SORT_KEY
from fbai.research.understat_xg.alignment import ALIGNED_XG_COLUMNS

UNDERSTAT_XG_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{name}{window}_{side}_xg_pre"
    for side in ("H", "A")
    for window in (5, 10)
    for name in ("XgForAvg", "XgAgainstAvg", "XgDiffAvg", "GminusXgForAvg")
)
UNDERSTAT_XG_FEATURE_TABLE_COLUMNS: tuple[str, ...] = (
    *NATURAL_KEY,
    *UNDERSTAT_XG_FEATURE_COLUMNS,
)

assert len(UNDERSTAT_XG_FEATURE_COLUMNS) == 16
assert len(UNDERSTAT_XG_FEATURE_COLUMNS) == len(set(UNDERSTAT_XG_FEATURE_COLUMNS))
assert all(column.endswith("_pre") for column in UNDERSTAT_XG_FEATURE_COLUMNS)


class UnderstatXGFeatureError(ValueError):
    """Raised when xG history or output violates the experimental contract."""


def validate_understat_xg_feature_table(
    frame: pd.DataFrame,
    *,
    expected_row_count: int | None = None,
) -> None:
    """Validate the exact keyed 16-feature table."""

    if tuple(frame.columns) != UNDERSTAT_XG_FEATURE_TABLE_COLUMNS:
        raise UnderstatXGFeatureError("Understat xG feature table has invalid columns or order")
    if expected_row_count is not None and len(frame) != expected_row_count:
        raise UnderstatXGFeatureError("Understat xG feature construction did not preserve rows")
    assert_natural_key_valid(frame, key_columns=NATURAL_KEY)
    assert_model_inputs_safe(
        UNDERSTAT_XG_FEATURE_COLUMNS,
        approved_pre_features=UNDERSTAT_XG_FEATURE_COLUMNS,
    )
    for column in UNDERSTAT_XG_FEATURE_COLUMNS:
        if not is_numeric_dtype(frame[column].dtype):
            raise UnderstatXGFeatureError(f"Understat xG feature {column} must be numeric")
    values = frame.loc[:, list(UNDERSTAT_XG_FEATURE_COLUMNS)].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if np.isinf(values).any():
        raise UnderstatXGFeatureError("Understat xG features contain infinite values")


def _validated_targets(target_matches: pd.DataFrame) -> pd.DataFrame:
    required = (*NATURAL_KEY, "SeasonStartYear")
    missing = [column for column in required if column not in target_matches.columns]
    if missing:
        raise UnderstatXGFeatureError(f"Missing Understat xG target columns: {', '.join(missing)}")
    targets = target_matches.loc[:, list(required)].copy(deep=True)
    targets["MatchDate"] = pd.to_datetime(targets["MatchDate"], errors="raise").dt.normalize()
    targets["SeasonStartYear"] = pd.to_numeric(targets["SeasonStartYear"], errors="raise").astype(
        int
    )
    assert_natural_key_valid(targets, key_columns=NATURAL_KEY)
    return targets.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(drop=True)


def _validated_history(aligned_xg_matches: pd.DataFrame) -> pd.DataFrame:
    if tuple(aligned_xg_matches.columns) != ALIGNED_XG_COLUMNS:
        raise UnderstatXGFeatureError("aligned xG history has invalid columns or order")
    history = aligned_xg_matches.copy(deep=True)
    history["MatchDate"] = pd.to_datetime(history["MatchDate"], errors="raise").dt.normalize()
    history["SeasonStartYear"] = pd.to_numeric(history["SeasonStartYear"], errors="raise").astype(
        int
    )
    assert_natural_key_valid(history, key_columns=NATURAL_KEY)
    for column in ("FTHome", "FTAway", "home_xg", "away_xg"):
        history[column] = pd.to_numeric(history[column], errors="raise").astype("float64")
    values = history.loc[:, ["home_xg", "away_xg"]].to_numpy(dtype=np.float64)
    one_side_missing = np.isnan(values).sum(axis=1) == 1
    present = values[~np.isnan(values)]
    if one_side_missing.any() or np.isinf(present).any() or (present < 0.0).any():
        raise UnderstatXGFeatureError(
            "present aligned xG history must be paired, finite, and non-negative"
        )
    return history.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(drop=True)


def build_understat_xg_feature_table(
    target_matches: pd.DataFrame,
    aligned_xg_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Build 16 source-defined features from strictly earlier same-season rows.

    Each calendar date is an indivisible batch because the lookup uses
    ``bisect_left`` on dates. The target row, other same-date rows, and future
    rows therefore cannot contribute to the target's state.
    """

    targets = _validated_targets(target_matches)
    history = _validated_history(aligned_xg_matches)

    # (date, season, xG for, xG against, goals for)
    team_history: dict[tuple[str, str], list[tuple[pd.Timestamp, int, float, float, float]]] = {}
    for row in history.itertuples(index=False):
        team_history.setdefault((str(row.Division), str(row.HomeTeam)), []).append(
            (
                pd.Timestamp(row.MatchDate),
                int(row.SeasonStartYear),
                float(row.home_xg),
                float(row.away_xg),
                float(row.FTHome),
            )
        )
        team_history.setdefault((str(row.Division), str(row.AwayTeam)), []).append(
            (
                pd.Timestamp(row.MatchDate),
                int(row.SeasonStartYear),
                float(row.away_xg),
                float(row.home_xg),
                float(row.FTAway),
            )
        )
    history_dates = {key: [entry[0] for entry in entries] for key, entries in team_history.items()}

    values = np.full((len(targets), len(UNDERSTAT_XG_FEATURE_COLUMNS)), np.nan)
    column_index = {column: index for index, column in enumerate(UNDERSTAT_XG_FEATURE_COLUMNS)}
    for output_row, target in enumerate(targets.itertuples(index=False)):
        for side, team in (("H", target.HomeTeam), ("A", target.AwayTeam)):
            key = (str(target.Division), str(team))
            if key not in team_history:
                continue
            position = bisect_left(history_dates[key], pd.Timestamp(target.MatchDate))
            past = [
                entry
                for entry in team_history[key][max(0, position - 10) : position]
                if entry[1] == int(target.SeasonStartYear)
            ]
            for window in (5, 10):
                selected = [
                    entry
                    for entry in past[-window:]
                    if not (np.isnan(entry[2]) or np.isnan(entry[3]))
                ]
                if not selected:
                    continue
                xg_for = float(np.mean([entry[2] for entry in selected]))
                xg_against = float(np.mean([entry[3] for entry in selected]))
                finishing = float(np.mean([entry[4] - entry[2] for entry in selected]))
                values[
                    output_row,
                    column_index[f"XgForAvg{window}_{side}_xg_pre"],
                ] = xg_for
                values[
                    output_row,
                    column_index[f"XgAgainstAvg{window}_{side}_xg_pre"],
                ] = xg_against
                values[
                    output_row,
                    column_index[f"XgDiffAvg{window}_{side}_xg_pre"],
                ] = xg_for - xg_against
                values[
                    output_row,
                    column_index[f"GminusXgForAvg{window}_{side}_xg_pre"],
                ] = finishing

    output = pd.concat(
        [
            targets.loc[:, list(NATURAL_KEY)].reset_index(drop=True),
            pd.DataFrame(values, columns=UNDERSTAT_XG_FEATURE_COLUMNS),
        ],
        axis=1,
    )
    validate_understat_xg_feature_table(output, expected_row_count=len(targets))
    return output
