"""Closed schema and timing audit for player-availability research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

FIXTURE_COLUMNS: tuple[str, ...] = (
    "Division",
    "SeasonStartYear",
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "game_id",
    "home_club_id",
    "away_club_id",
)
APPEARANCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "player_club_id",
    "minutes_played",
)
LINEUP_COLUMNS: tuple[str, ...] = (
    "game_id",
    "player_id",
    "club_id",
    "type",
)
VALUATION_COLUMNS: tuple[str, ...] = (
    "player_id",
    "date",
    "market_value_in_eur",
)
ALLOWED_LINEUP_ROLES: frozenset[str] = frozenset({"starting_lineup", "substitutes"})

PRIOR_BASE_FEATURES: tuple[str, ...] = (
    "matches_30d",
    "minutes_7d",
    "minutes_14d",
    "minutes_21d",
    "minutes_30d",
    "days_since_prev_match",
    "unique_players_5",
    "unique_players_10",
    "unique_starters_5",
    "top5_minutes_share_5",
    "top11_minutes_share_5",
    "top11_minutes_share_10",
    "avg_player_minutes_5",
    "starter_overlap_last",
    "starter_jaccard_last",
    "rotation_index_last",
    "avg_starter_changes_5",
    "recent_players_value_sum_5",
    "recent_top11_value_sum_5",
    "last_starters_value_sum",
    "recent_value_coverage_5",
)

PRIOR_AVAILABILITY_FEATURES: tuple[str, ...] = (
    *(f"avail_{side}_{feature}_pre" for side in ("H", "A") for feature in PRIOR_BASE_FEATURES),
    *(f"avail_diff_{feature}_pre" for feature in PRIOR_BASE_FEATURES),
)


class InformationTiming(StrEnum):
    """The four timing classes required by the Phase 5C3 audit."""

    KNOWN_BEFORE_KICKOFF = "known_before_kickoff"
    COMPLETED_PRIOR_MATCH_ONLY = "derived_only_from_completed_prior_matches"
    KNOWN_ONLY_AFTER_TARGET_MATCH = "known_only_after_target_match"
    TIMING_UNKNOWN = "timing_unknown"


@dataclass(frozen=True, slots=True)
class TimingClassification:
    """One source field group's timing and public-use decision."""

    source_group: str
    fields: tuple[str, ...]
    timing: InformationTiming
    public_target_use: str
    reason: str

    def to_dict(self) -> dict[str, str | list[str]]:
        return {
            "source_group": self.source_group,
            "fields": list(self.fields),
            "timing": self.timing.value,
            "public_target_use": self.public_target_use,
            "reason": self.reason,
        }


TIMING_AUDIT: tuple[TimingClassification, ...] = (
    TimingClassification(
        source_group="fixture_identity",
        fields=FIXTURE_COLUMNS,
        timing=InformationTiming.KNOWN_BEFORE_KICKOFF,
        public_target_use="join_and_chronology_only",
        reason="Fixture identity and scheduled date are not model features.",
    ),
    TimingClassification(
        source_group="appearances",
        fields=APPEARANCE_COLUMNS,
        timing=InformationTiming.COMPLETED_PRIOR_MATCH_ONLY,
        public_target_use="prior_matches_only",
        reason=(
            "Participation and minutes are known after a match; target and same-date "
            "appearance rows are excluded."
        ),
    ),
    TimingClassification(
        source_group="lineups_historical",
        fields=LINEUP_COLUMNS,
        timing=InformationTiming.COMPLETED_PRIOR_MATCH_ONLY,
        public_target_use="prior_matches_only",
        reason="Completed prior-match starter roles support continuity features.",
    ),
    TimingClassification(
        source_group="target_official_lineup_and_bench",
        fields=("starting_lineup", "substitutes"),
        timing=InformationTiming.TIMING_UNKNOWN,
        public_target_use="forbidden",
        reason=(
            "The local source has no publication timestamp proving that target rows "
            "were available before kickoff."
        ),
    ),
    TimingClassification(
        source_group="player_valuations",
        fields=VALUATION_COLUMNS,
        timing=InformationTiming.KNOWN_BEFORE_KICKOFF,
        public_target_use="only_when_valuation_date_is_strictly_earlier",
        reason="Same-date and future valuations are excluded by left-bound lookup.",
    ),
    TimingClassification(
        source_group="target_participation_and_substitutions",
        fields=("minutes_played", "appearance", "substitution"),
        timing=InformationTiming.KNOWN_ONLY_AFTER_TARGET_MATCH,
        public_target_use="forbidden",
        reason="These fields describe in-match or completed-match events.",
    ),
    TimingClassification(
        source_group="injury_suspension_absence_labels",
        fields=("injury", "suspension", "absence_start", "absence_end"),
        timing=InformationTiming.TIMING_UNKNOWN,
        public_target_use="not_available",
        reason="The authoritative local artifact contains no clean absence-window table.",
    ),
)


@dataclass(frozen=True, slots=True)
class AvailabilityClassifierConfig:
    """Exact source Logistic Regression configuration for LR52+63."""

    solver: str = "lbfgs"
    penalty: str = "l2"
    regularization_c: float = 1.0
    max_iter: int = 2000
    fit_intercept: bool = True
    class_weight: str | None = None
    random_seed: int = 42
    tolerance: float = 1e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.solver != "lbfgs" or self.penalty != "l2":
            raise ValueError("availability classifier is fixed to lbfgs L2")
        if self.regularization_c <= 0.0 or self.max_iter < 1 or self.tolerance <= 0.0:
            raise ValueError("availability classifier optimization settings are invalid")
        if self.class_weight is not None:
            raise ValueError("availability classifier class_weight is fixed to None")
        if self.device != "cpu":
            raise ValueError("availability research supports CPU execution only")

    @property
    def identifier(self) -> str:
        return "lr115_median_standardized_lbfgs_l2_c1_iter2000_seed42"

    @property
    def input_feature_count(self) -> int:
        return 52 + len(PRIOR_AVAILABILITY_FEATURES)

    @property
    def parameter_count(self) -> int:
        return 3 * (self.input_feature_count + 1)

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        values: dict[str, str | int | float | bool | None] = asdict(self)
        values["identifier"] = self.identifier
        values["input_feature_count"] = self.input_feature_count
        values["parameter_count"] = self.parameter_count
        return values


AVAILABILITY_CLASSIFIER_CONFIG = AvailabilityClassifierConfig()

assert len(PRIOR_BASE_FEATURES) == 21
assert len(PRIOR_AVAILABILITY_FEATURES) == 63
assert AVAILABILITY_CLASSIFIER_CONFIG.input_feature_count == 115
assert AVAILABILITY_CLASSIFIER_CONFIG.parameter_count == 348
