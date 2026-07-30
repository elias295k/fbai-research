from __future__ import annotations

import pandas as pd
import pytest

from fbai.data.schema import NATURAL_KEY
from fbai.research.understat_xg.alignment import ALIGNED_XG_COLUMNS
from fbai.research.understat_xg.features import (
    UNDERSTAT_XG_FEATURE_COLUMNS,
    build_understat_xg_feature_table,
)


def _history() -> pd.DataFrame:
    rows = [
        ("2023-08-01", "A", "B", 1.0, 0.5, 1, 0),
        ("2023-08-08", "C", "A", 1.4, 0.8, 2, 1),
        ("2023-08-15", "A", "D", 2.0, 1.0, 3, 1),
        ("2023-08-22", "B", "A", 0.7, 1.5, 0, 2),
        ("2023-08-22", "C", "D", 1.2, 1.1, 1, 1),
        ("2023-08-29", "A", "C", 1.8, 0.9, 2, 0),
    ]
    return pd.DataFrame(
        [
            {
                "MatchDate": pd.Timestamp(date),
                "Division": "E0",
                "HomeTeam": home,
                "AwayTeam": away,
                "SeasonStartYear": 2023,
                "FTHome": home_goals,
                "FTAway": away_goals,
                "home_xg": home_xg,
                "away_xg": away_xg,
            }
            for date, home, away, home_xg, away_xg, home_goals, away_goals in rows
        ],
        columns=ALIGNED_XG_COLUMNS,
    )


def _targets(history: pd.DataFrame) -> pd.DataFrame:
    return history.loc[:, [*NATURAL_KEY, "SeasonStartYear"]].copy()


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index(list(NATURAL_KEY)).sort_index()


def test_exact_feature_tuple_and_partial_same_season_windows() -> None:
    history = _history()
    features = build_understat_xg_feature_table(_targets(history), history)
    last = features.iloc[-1]

    assert len(UNDERSTAT_XG_FEATURE_COLUMNS) == 16
    assert tuple(features.columns) == (*NATURAL_KEY, *UNDERSTAT_XG_FEATURE_COLUMNS)
    assert last["XgForAvg5_H_xg_pre"] == pytest.approx(1.325)
    assert last["XgAgainstAvg5_H_xg_pre"] == pytest.approx(0.9)
    assert last["XgDiffAvg5_H_xg_pre"] == pytest.approx(0.425)
    assert last["GminusXgForAvg5_H_xg_pre"] == pytest.approx(0.425)


def test_future_and_current_xg_cannot_change_past_or_own_features() -> None:
    history = _history()
    targets = _targets(history)
    original = build_understat_xg_feature_table(targets, history)
    changed = history.copy(deep=True)
    changed.loc[changed.index[-1], ["home_xg", "away_xg"]] = [8.0, 7.0]
    rebuilt = build_understat_xg_feature_table(targets, changed)

    pd.testing.assert_series_equal(original.iloc[-1], rebuilt.iloc[-1])
    pd.testing.assert_frame_equal(original.iloc[:-1], rebuilt.iloc[:-1])


def test_same_date_is_an_indivisible_information_batch() -> None:
    history = _history()
    target = _targets(history).loc[[4]]
    original = build_understat_xg_feature_table(target, history)
    changed = history.copy(deep=True)
    changed.loc[3, ["home_xg", "away_xg"]] = [8.0, 7.0]
    rebuilt = build_understat_xg_feature_table(target, changed)

    pd.testing.assert_frame_equal(original, rebuilt)


def test_shuffled_inputs_produce_identical_keyed_output() -> None:
    history = _history()
    targets = _targets(history)
    expected = build_understat_xg_feature_table(targets, history)
    actual = build_understat_xg_feature_table(
        targets.sample(frac=1.0, random_state=4),
        history.sample(frac=1.0, random_state=8),
    )

    pd.testing.assert_frame_equal(_keyed(expected), _keyed(actual))


def test_season_reset_prevents_carryover() -> None:
    history = _history()
    next_season = pd.DataFrame(
        [
            {
                "MatchDate": pd.Timestamp("2024-08-01"),
                "Division": "E0",
                "HomeTeam": "A",
                "AwayTeam": "B",
                "SeasonStartYear": 2024,
            }
        ]
    )

    features = build_understat_xg_feature_table(next_season, history)

    assert features.loc[:, list(UNDERSTAT_XG_FEATURE_COLUMNS)].isna().all().all()
