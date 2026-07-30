"""Team-perspective, strictly-prior rolling pseudo-xG feature construction."""

from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fbai.core.leakage import assert_model_inputs_safe, assert_natural_key_valid
from fbai.data.schema import NATURAL_KEY, STABLE_SORT_KEY, validate_canonical_frame
from fbai.research.pseudo_xg.config import PseudoXGConfig
from fbai.research.pseudo_xg.model import (
    FittedPseudoXGEstimator,
    fit_pseudo_xg_estimator,
    predict_match_pseudo_xg,
)

PSEUDO_XG_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"{name}{window}_{side}_pxg_pre"
    for side in ("H", "A")
    for window in (5, 10)
    for name in ("PxgForAvg", "PxgAgainstAvg", "GminusPxgForAvg")
)
PSEUDO_XG_TABLE_COLUMNS: tuple[str, ...] = (*NATURAL_KEY, *PSEUDO_XG_FEATURE_COLUMNS)

assert len(PSEUDO_XG_FEATURE_COLUMNS) == 12
assert len(PSEUDO_XG_FEATURE_COLUMNS) == len(set(PSEUDO_XG_FEATURE_COLUMNS))
assert all(column.endswith("_pre") for column in PSEUDO_XG_FEATURE_COLUMNS)


class PseudoXGFeatureError(ValueError):
    """Raised when pseudo-xG feature history violates the research contract."""


def validate_pseudo_xg_feature_table(
    frame: pd.DataFrame,
    *,
    expected_row_count: int | None = None,
) -> None:
    """Validate the exact keyed, aggregate pseudo-xG feature table."""

    if tuple(frame.columns) != PSEUDO_XG_TABLE_COLUMNS:
        raise PseudoXGFeatureError("pseudo-xG feature table has invalid columns or order")
    if expected_row_count is not None and len(frame) != expected_row_count:
        raise PseudoXGFeatureError("pseudo-xG feature construction did not preserve rows")
    assert_natural_key_valid(frame, key_columns=NATURAL_KEY)
    assert_model_inputs_safe(
        PSEUDO_XG_FEATURE_COLUMNS,
        approved_pre_features=PSEUDO_XG_FEATURE_COLUMNS,
    )
    for column in PSEUDO_XG_FEATURE_COLUMNS:
        if not is_numeric_dtype(frame[column].dtype):
            raise PseudoXGFeatureError(f"pseudo-xG feature {column} must be numeric")
    values = frame.loc[:, list(PSEUDO_XG_FEATURE_COLUMNS)].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if np.isinf(values).any():
        raise PseudoXGFeatureError("pseudo-xG features contain infinite values")


def _validated_targets(target_matches: pd.DataFrame) -> pd.DataFrame:
    required = (*NATURAL_KEY, "SeasonStartYear")
    missing = sorted(set(required).difference(target_matches.columns))
    if missing:
        raise PseudoXGFeatureError(f"Missing pseudo-xG target columns: {', '.join(missing)}")
    targets = target_matches.loc[:, list(required)].copy(deep=True)
    targets["MatchDate"] = pd.to_datetime(targets["MatchDate"], errors="raise")
    targets["SeasonStartYear"] = pd.to_numeric(targets["SeasonStartYear"], errors="raise").astype(
        int
    )
    assert_natural_key_valid(targets, key_columns=NATURAL_KEY)
    return targets.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(drop=True)


def build_pseudo_xg_feature_table(
    target_matches: pd.DataFrame,
    historical_matches: pd.DataFrame,
    fitted: FittedPseudoXGEstimator,
) -> pd.DataFrame:
    """Build source-defined features from completed matches strictly before each date.

    The estimator is already fold-local. This builder treats every calendar
    date as an indivisible information batch, so current, same-date, and future
    match statistics never enter a target row's rolling history.
    """

    targets = _validated_targets(target_matches)
    validate_canonical_frame(historical_matches)
    history = historical_matches.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(
        drop=True
    )
    history_keys = pd.MultiIndex.from_frame(history.loc[:, list(NATURAL_KEY)])
    target_keys = pd.MultiIndex.from_frame(targets.loc[:, list(NATURAL_KEY)])
    if len(target_keys.difference(history_keys)):
        raise PseudoXGFeatureError("pseudo-xG target keys must exist in canonical history")

    home_pxg = predict_match_pseudo_xg(
        fitted,
        shots_on_target=history["HomeTarget"],
        shots=history["HomeShots"],
    )
    away_pxg = predict_match_pseudo_xg(
        fitted,
        shots_on_target=history["AwayTarget"],
        shots=history["AwayShots"],
    )
    history = history.assign(PseudoXGHome=home_pxg, PseudoXGAway=away_pxg)

    # (date, season, pseudo-xG for, pseudo-xG against, goals for)
    team_history: dict[tuple[str, str], list[tuple[pd.Timestamp, int, float, float, float]]] = {}
    for row in history.itertuples(index=False):
        home_key = (str(row.Division), str(row.HomeTeam))
        away_key = (str(row.Division), str(row.AwayTeam))
        team_history.setdefault(home_key, []).append(
            (
                pd.Timestamp(row.MatchDate),
                int(row.SeasonStartYear),
                float(row.PseudoXGHome),
                float(row.PseudoXGAway),
                float(row.FTHome),
            )
        )
        team_history.setdefault(away_key, []).append(
            (
                pd.Timestamp(row.MatchDate),
                int(row.SeasonStartYear),
                float(row.PseudoXGAway),
                float(row.PseudoXGHome),
                float(row.FTAway),
            )
        )
    history_dates = {key: [entry[0] for entry in entries] for key, entries in team_history.items()}

    values = np.full((len(targets), len(PSEUDO_XG_FEATURE_COLUMNS)), np.nan)
    column_index = {name: index for index, name in enumerate(PSEUDO_XG_FEATURE_COLUMNS)}
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
            for window in fitted.config.rolling_windows:
                valid = [
                    (pxg_for, pxg_against, goals_for)
                    for _, _, pxg_for, pxg_against, goals_for in past[-window:]
                    if not (np.isnan(pxg_for) or np.isnan(pxg_against))
                ]
                if not valid:
                    continue
                pxg_for_mean = float(np.mean([entry[0] for entry in valid]))
                pxg_against_mean = float(np.mean([entry[1] for entry in valid]))
                finishing_mean = float(np.mean([entry[2] - entry[0] for entry in valid]))
                values[
                    output_row,
                    column_index[f"PxgForAvg{window}_{side}_pxg_pre"],
                ] = pxg_for_mean
                values[
                    output_row,
                    column_index[f"PxgAgainstAvg{window}_{side}_pxg_pre"],
                ] = pxg_against_mean
                values[
                    output_row,
                    column_index[f"GminusPxgForAvg{window}_{side}_pxg_pre"],
                ] = finishing_mean

    output = pd.concat(
        [
            targets.loc[:, list(NATURAL_KEY)].reset_index(drop=True),
            pd.DataFrame(values, columns=PSEUDO_XG_FEATURE_COLUMNS),
        ],
        axis=1,
    )
    validate_pseudo_xg_feature_table(output, expected_row_count=len(targets))
    return output


def build_walk_forward_training_feature_table(
    training_matches: pd.DataFrame,
    *,
    config: PseudoXGConfig | None = None,
) -> pd.DataFrame:
    """Build leakage-safe candidate-training features with season-boundary refits.

    The authority used one outer-train estimator to transform that same outer
    training partition. Because its goals and shots could then affect earlier
    transformed training rows through the learned coefficients, the public
    interpretation refits only on matches before each season's first date.
    This preserves the source formula and rolling features while making current,
    same-date, and future training rows unable to affect an earlier row.
    """

    validate_canonical_frame(training_matches)
    ordered = training_matches.sort_values(list(STABLE_SORT_KEY), kind="mergesort").reset_index(
        drop=True
    )
    outputs: list[pd.DataFrame] = []
    for season in sorted(ordered["SeasonStartYear"].astype(int).unique()):
        season_targets = ordered.loc[ordered["SeasonStartYear"].eq(season)]
        first_target_date = pd.Timestamp(season_targets["MatchDate"].min())
        estimator_train = ordered.loc[ordered["MatchDate"].lt(first_target_date)]
        if estimator_train.empty:
            empty = season_targets.loc[:, list(NATURAL_KEY)].reset_index(drop=True)
            for column in PSEUDO_XG_FEATURE_COLUMNS:
                empty[column] = np.nan
            outputs.append(empty.loc[:, list(PSEUDO_XG_TABLE_COLUMNS)])
            continue
        fitted = fit_pseudo_xg_estimator(estimator_train, config=config)
        outputs.append(
            build_pseudo_xg_feature_table(
                season_targets,
                ordered,
                fitted,
            )
        )
    output = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(list(STABLE_SORT_KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    validate_pseudo_xg_feature_table(output, expected_row_count=len(ordered))
    return output
