from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbai.research.understat_xg.alignment import (
    UnderstatXGAlignmentError,
    align_understat_xg,
)
from fbai.testing.synthetic import make_synthetic_canonical_matches


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = make_synthetic_canonical_matches(
        seed=520,
        season_start_years=(2020, 2021, 2022, 2023, 2024, 2025),
        divisions=("E0", "D1"),
        teams_per_division=6,
    )
    external = canonical.loc[
        canonical["SeasonStartYear"].le(2023),
        ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"],
    ].copy()
    index = np.arange(len(external))
    noise_home = ((index * 7) % 7 - 3) * 0.18
    noise_away = ((index * 11) % 7 - 3) * 0.18
    external["home_xg"] = np.clip(
        1.05 + 0.38 * external["FTHome"].to_numpy() + noise_home,
        0.05,
        None,
    )
    external["away_xg"] = np.clip(
        1.05 + 0.38 * external["FTAway"].to_numpy() + noise_away,
        0.05,
        None,
    )
    return canonical, external.drop(columns=["FTHome", "FTAway"]).reset_index(drop=True)


def test_exact_one_to_one_alignment_reports_all_coverage_views() -> None:
    canonical, external = _inputs()
    result = align_understat_xg(canonical, external)
    report = result.report

    assert report.passed
    assert report.covered_divisions == ("D1", "E0")
    assert report.aligned_rows == len(external)
    assert report.unmatched_external_rows == 0
    assert report.unmatched_canonical_rows == len(canonical) - len(external)
    assert len(report.coverage_by_division) == 2
    assert len(report.coverage_by_league_season) == 12
    assert [record.name for record in report.coverage_by_evaluation_fold] == [
        "test_2022",
        "test_2023",
        "test_2024",
        "test_2025",
    ]


def test_source_alias_can_align_without_fuzzy_matching() -> None:
    canonical, external = _inputs()
    canonical.loc[0, "HomeTeam"] = "Man United"
    external.loc[0, "HomeTeam"] = "Manchester United"

    result = align_understat_xg(canonical, external)

    assert result.report.unmatched_external_rows == 0
    assert result.report.method.startswith("one_to_one exact")


def test_unmatched_external_and_canonical_rows_are_aggregate_only() -> None:
    canonical, external = _inputs()
    external.loc[0, "MatchDate"] += pd.Timedelta(days=1)

    result = align_understat_xg(canonical, external, require_quality_pass=False)

    assert result.report.unmatched_external_rows == 1
    assert result.report.unmatched_canonical_rows == len(canonical) - len(external) + 1
    assert result.report.to_dict()["aligned_rows"] == len(external) - 1


def test_quality_gate_rejects_implausible_xg() -> None:
    canonical, external = _inputs()
    external["home_xg"] = 9.0
    external["away_xg"] = 9.0

    with pytest.raises(UnderstatXGAlignmentError, match="quality gates"):
        align_understat_xg(canonical, external)
