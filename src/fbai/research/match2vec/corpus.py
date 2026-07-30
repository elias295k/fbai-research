"""Deterministic team vocabulary and strictly pre-match descriptor sequences."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from fbai.core.leakage import assert_natural_key_valid
from fbai.data.schema import STABLE_SORT_KEY, validate_canonical_frame

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "was_home",
    "goals_for",
    "goals_against",
    "goal_difference",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "points",
    "log_days_ago",
    "same_season",
)
DESCRIPTOR_COUNT = len(DESCRIPTOR_NAMES)
MatchKey: TypeAlias = tuple[pd.Timestamp, str, str, str]
TeamToken: TypeAlias = tuple[str, str]


def _key(row: Any) -> MatchKey:
    return (
        pd.Timestamp(row.MatchDate),
        str(row.Division),
        str(row.HomeTeam),
        str(row.AwayTeam),
    )


def _validate_key_frame(frame: pd.DataFrame) -> None:
    assert_natural_key_valid(
        frame,
        key_columns=("Division", "MatchDate", "HomeTeam", "AwayTeam"),
    )
    if frame.empty:
        raise ValueError("Match sequence input must contain at least one row")


@dataclass(frozen=True, slots=True)
class Match2VecVocabulary:
    """Fold-local deterministic team/league vocabulary with per-league UNK."""

    team_to_id: MappingProxyType[TeamToken, int]
    league_to_id: MappingProxyType[str, int]
    league_unknown_id: MappingProxyType[str, int]

    @property
    def team_token_count(self) -> int:
        return len(self.team_to_id)

    @property
    def unknown_token_count(self) -> int:
        return len(self.league_unknown_id)

    @property
    def total_team_token_count(self) -> int:
        return self.team_token_count + self.unknown_token_count

    @property
    def league_count(self) -> int:
        return len(self.league_to_id)

    def _unknown_for(self, division: str) -> int:
        try:
            return self.league_unknown_id[division]
        except KeyError as exc:
            raise ValueError(
                f"Division {division!r} was absent from the training vocabulary"
            ) from exc

    def team_ids(
        self,
        frame: pd.DataFrame,
        *,
        side: str,
    ) -> npt.NDArray[np.int64]:
        """Map a side to training IDs, using its league-specific unknown token."""

        if side not in {"home", "away"}:
            raise ValueError("side must be 'home' or 'away'")
        column = "HomeTeam" if side == "home" else "AwayTeam"
        values = np.empty(len(frame), dtype=np.int64)
        for position, (division, team) in enumerate(
            zip(frame["Division"].astype(str), frame[column].astype(str), strict=True)
        ):
            values[position] = self.team_to_id.get(
                (division, team),
                self._unknown_for(division),
            )
        return values

    def league_ids(self, frame: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Map divisions, rejecting a division never observed in the training fold."""

        values: list[int] = []
        for division in frame["Division"].astype(str):
            if division not in self.league_to_id:
                raise ValueError(f"Division {division!r} was absent from the training vocabulary")
            values.append(self.league_to_id[division])
        return np.asarray(values, dtype=np.int64)

    def oov_counts(self, frame: pd.DataFrame) -> tuple[int, int]:
        """Return home and away occurrences not represented by a training team token."""

        home = sum(
            (str(division), str(team)) not in self.team_to_id
            for division, team in zip(frame["Division"], frame["HomeTeam"], strict=True)
        )
        away = sum(
            (str(division), str(team)) not in self.team_to_id
            for division, team in zip(frame["Division"], frame["AwayTeam"], strict=True)
        )
        return home, away


def build_train_vocabulary(train_rows: pd.DataFrame) -> Match2VecVocabulary:
    """Build stable team and league IDs from the current outer training fold only."""

    _validate_key_frame(train_rows)
    leagues = sorted(str(value) for value in train_rows["Division"].unique())
    league_to_id = {division: index for index, division in enumerate(leagues)}
    pairs = sorted(
        {
            (str(division), str(team))
            for division, team in zip(
                train_rows["Division"],
                train_rows["HomeTeam"],
                strict=True,
            )
        }
        | {
            (str(division), str(team))
            for division, team in zip(
                train_rows["Division"],
                train_rows["AwayTeam"],
                strict=True,
            )
        }
    )
    team_to_id = {token: index for index, token in enumerate(pairs)}
    league_unknown_id = {
        division: len(team_to_id) + index for index, division in enumerate(leagues)
    }
    return Match2VecVocabulary(
        team_to_id=MappingProxyType(team_to_id),
        league_to_id=MappingProxyType(league_to_id),
        league_unknown_id=MappingProxyType(league_unknown_id),
    )


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Read-only sequence tensors aligned to an explicit tuple of natural keys."""

    keys: tuple[MatchKey, ...]
    home_sequences: npt.NDArray[np.float32]
    home_mask: npt.NDArray[np.bool_]
    away_sequences: npt.NDArray[np.float32]
    away_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        arrays = {
            "home_sequences": np.asarray(self.home_sequences, dtype=np.float32).copy(),
            "home_mask": np.asarray(self.home_mask, dtype=bool).copy(),
            "away_sequences": np.asarray(self.away_sequences, dtype=np.float32).copy(),
            "away_mask": np.asarray(self.away_mask, dtype=bool).copy(),
        }
        expected_rows = len(self.keys)
        if arrays["home_sequences"].shape != arrays["away_sequences"].shape:
            raise ValueError("Home and away sequence shapes must match")
        if arrays["home_sequences"].ndim != 3:
            raise ValueError("Sequence tensors must have shape (rows, window, descriptors)")
        if arrays["home_sequences"].shape[0] != expected_rows:
            raise ValueError("Sequence row count does not match natural keys")
        expected_mask = arrays["home_sequences"].shape[:2]
        if arrays["home_mask"].shape != expected_mask or arrays["away_mask"].shape != expected_mask:
            raise ValueError("Sequence masks do not match their sequence tensors")
        if arrays["home_sequences"].shape[2] != DESCRIPTOR_COUNT:
            raise ValueError("Sequence descriptor count does not match the source contract")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def row_count(self) -> int:
        return len(self.keys)

    @property
    def sequence_length(self) -> int:
        return int(self.home_sequences.shape[1])


class MatchSequenceCorpus:
    """Key-addressable immutable collection of pre-match sequence descriptors."""

    def __init__(self, batch: SequenceBatch):
        if len(batch.keys) != len(set(batch.keys)):
            raise ValueError("Sequence corpus natural keys must be unique")
        self._batch = batch
        self._position = MappingProxyType(
            {key: position for position, key in enumerate(batch.keys)}
        )

    @property
    def keys(self) -> tuple[MatchKey, ...]:
        return self._batch.keys

    @property
    def row_count(self) -> int:
        return self._batch.row_count

    @property
    def sequence_length(self) -> int:
        return self._batch.sequence_length

    def batch_for(self, rows: pd.DataFrame) -> SequenceBatch:
        """Return sequences in the requested row order, aligned by natural key."""

        _validate_key_frame(rows)
        keys = tuple(_key(row) for row in rows.itertuples(index=False))
        missing = [key for key in keys if key not in self._position]
        if missing:
            raise ValueError(f"Sequence corpus is missing {len(missing)} requested natural keys")
        positions = np.asarray([self._position[key] for key in keys], dtype=np.int64)
        return SequenceBatch(
            keys=keys,
            home_sequences=self._batch.home_sequences[positions],
            home_mask=self._batch.home_mask[positions],
            away_sequences=self._batch.away_sequences[positions],
            away_mask=self._batch.away_mask[positions],
        )


def _scaled(value: object, scale: float) -> float:
    return 0.0 if pd.isna(value) else float(cast(Any, value)) / scale


def _team_descriptor(row: Any, team: str) -> list[float]:
    home = str(row.HomeTeam) == team
    home_goals = row.FTHome
    away_goals = row.FTAway
    goals_for = home_goals if home else away_goals
    goals_against = away_goals if home else home_goals
    shots_for = row.HomeShots if home else row.AwayShots
    shots_against = row.AwayShots if home else row.HomeShots
    target_for = row.HomeTarget if home else row.AwayTarget
    target_against = row.AwayTarget if home else row.HomeTarget
    corners_for = row.HomeCorners if home else row.AwayCorners
    corners_against = row.AwayCorners if home else row.HomeCorners
    result = str(row.FTR)
    won = result == ("H" if home else "A")
    points = 3.0 if won else (1.0 if result == "D" else 0.0)
    scaled_for = _scaled(goals_for, 3.0)
    scaled_against = _scaled(goals_against, 3.0)
    return [
        1.0 if home else 0.0,
        scaled_for,
        scaled_against,
        scaled_for - scaled_against,
        _scaled(shots_for, 15.0),
        _scaled(shots_against, 15.0),
        _scaled(target_for, 5.0),
        _scaled(target_against, 5.0),
        _scaled(corners_for, 6.0),
        _scaled(corners_against, 6.0),
        points / 3.0,
        0.0,
        0.0,
    ]


def build_match_sequence_corpus(
    target_rows: pd.DataFrame,
    historical_matches: pd.DataFrame,
    *,
    sequence_length: int = 10,
) -> MatchSequenceCorpus:
    """Build source-verified sequences from matches strictly before each target date.

    All rows on one date are an indivisible information batch because lookup
    uses ``bisect_left`` on dates. Stable natural-key sorting also makes output
    independent of input row order.
    """

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    _validate_key_frame(target_rows)
    validate_canonical_frame(historical_matches)
    ordered_history = historical_matches.sort_values(
        list(STABLE_SORT_KEY),
        kind="mergesort",
    ).reset_index(drop=True)
    ordered_targets = target_rows.copy(deep=True)
    ordered_targets["MatchDate"] = pd.to_datetime(ordered_targets["MatchDate"], errors="raise")
    ordered_targets = ordered_targets.sort_values(
        list(STABLE_SORT_KEY),
        kind="mergesort",
    ).reset_index(drop=True)

    history: dict[TeamToken, list[tuple[pd.Timestamp, int, list[float]]]] = {}
    for row in ordered_history.itertuples(index=False):
        for team in (str(row.HomeTeam), str(row.AwayTeam)):
            token = (str(row.Division), team)
            history.setdefault(token, []).append(
                (
                    pd.Timestamp(row.MatchDate),
                    int(row.SeasonStartYear),
                    _team_descriptor(row, team),
                )
            )
    history_dates = {token: [entry[0] for entry in entries] for token, entries in history.items()}

    row_count = len(ordered_targets)
    home_sequences = np.zeros(
        (row_count, sequence_length, DESCRIPTOR_COUNT),
        dtype=np.float32,
    )
    away_sequences = np.zeros_like(home_sequences)
    home_mask = np.zeros((row_count, sequence_length), dtype=bool)
    away_mask = np.zeros_like(home_mask)
    days_index = DESCRIPTOR_NAMES.index("log_days_ago")
    season_index = DESCRIPTOR_NAMES.index("same_season")

    for position, row in enumerate(ordered_targets.itertuples(index=False)):
        target_date = pd.Timestamp(row.MatchDate)
        for side, team, sequences, mask in (
            ("home", str(row.HomeTeam), home_sequences, home_mask),
            ("away", str(row.AwayTeam), away_sequences, away_mask),
        ):
            token = (str(row.Division), team)
            entries = history.get(token, [])
            prior_count = bisect_left(history_dates.get(token, []), target_date)
            prior = entries[max(0, prior_count - sequence_length) : prior_count]
            for sequence_position, (match_date, season, descriptor) in enumerate(prior):
                values = list(descriptor)
                values[days_index] = float(np.log1p((target_date - match_date).days)) / 7.0
                values[season_index] = 1.0 if season == int(row.SeasonStartYear) else 0.0
                sequences[position, sequence_position] = values
                mask[position, sequence_position] = True
            if side not in {"home", "away"}:  # pragma: no cover - tuple is closed above
                raise AssertionError("unreachable")

    keys = tuple(_key(row) for row in ordered_targets.itertuples(index=False))
    return MatchSequenceCorpus(
        SequenceBatch(
            keys=keys,
            home_sequences=home_sequences,
            home_mask=home_mask,
            away_sequences=away_sequences,
            away_mask=away_mask,
        )
    )
