from __future__ import annotations

import pandas as pd
import pytest

from fbai.data.audit import AuditFailure, audit_canonical
from fbai.testing.synthetic import make_synthetic_canonical_matches


def canonical() -> pd.DataFrame:
    return make_synthetic_canonical_matches(
        seed=27,
        season_start_years=(2023,),
        divisions=("SYN1",),
        teams_per_division=4,
    )


def test_clean_canonical_input_passes() -> None:
    frame = canonical()

    result = audit_canonical(frame, input_row_count=len(frame))

    assert result.passed
    assert result.rows_preserved
    assert result.number_of_divisions == 1
    assert result.number_of_seasons == 1
    assert result.issues == ()


def test_duplicate_rows_fail() -> None:
    frame = canonical()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    result = audit_canonical(duplicated)

    assert not result.passed
    assert result.duplicate_natural_keys > 0
    with pytest.raises(AuditFailure, match="duplicate natural keys"):
        result.raise_for_failure()


def test_invalid_labels_fail() -> None:
    frame = canonical()
    frame.loc[0, "FTR"] = "W"

    result = audit_canonical(frame)

    assert not result.passed
    assert result.invalid_target_labels == 1


def test_missing_input_rows_are_detected() -> None:
    frame = canonical()

    result = audit_canonical(frame.iloc[:-1], input_row_count=len(frame))

    assert not result.passed
    assert not result.rows_preserved
    assert result.input_row_count == len(frame)
    assert result.output_row_count == len(frame) - 1


def test_extra_output_rows_are_a_row_preservation_mismatch() -> None:
    frame = canonical()

    result = audit_canonical(frame, input_row_count=len(frame) - 1)

    assert not result.passed
    assert not result.rows_preserved


def test_invalid_dates_fail() -> None:
    frame = canonical()
    frame["MatchDate"] = frame["MatchDate"].astype(object)
    frame.loc[0, "MatchDate"] = "not-a-date"

    result = audit_canonical(frame)

    assert not result.passed
    assert result.invalid_dates == 1


def test_missing_required_columns_fail() -> None:
    result = audit_canonical(canonical().drop(columns="AwayFouls"))

    assert not result.passed
    assert result.missing_required_columns == ("AwayFouls",)


def test_unsorted_rows_fail_deterministic_ordering() -> None:
    frame = canonical().sample(frac=1.0, random_state=4).reset_index(drop=True)

    result = audit_canonical(frame)

    assert not result.passed
    assert not result.deterministic_ordering
