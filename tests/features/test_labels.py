from __future__ import annotations

import pandas as pd
import pytest

from fbai.features.labels import TARGET_COLUMNS, LabelValidationError, add_target_labels
from fbai.features.schema import FEATURE_COLUMNS


def test_valid_results_map_to_source_verified_targets() -> None:
    frame = pd.DataFrame(
        {
            "FTR": ["H", "D", "A"],
            "FTHome": [3, 1, 0],
            "FTAway": [0, 1, 4],
        }
    )

    labeled = add_target_labels(frame)

    assert labeled["target_1x2"].tolist() == ["H", "D", "A"]
    assert labeled["target_home_win"].tolist() == [1, 0, 0]
    assert labeled["target_draw"].tolist() == [0, 1, 0]
    assert labeled["target_away_win"].tolist() == [0, 0, 1]
    assert labeled["target_ou25"].tolist() == [1, 0, 1]
    assert tuple(column for column in TARGET_COLUMNS if column in FEATURE_COLUMNS) == ()


def test_missing_goals_preserve_missing_over_under_target() -> None:
    frame = pd.DataFrame({"FTR": ["H"], "FTHome": [pd.NA], "FTAway": [0]})

    labeled = add_target_labels(frame)

    assert pd.isna(labeled.loc[0, "target_ou25"])


@pytest.mark.parametrize("invalid", ["X", "", None])
def test_invalid_results_raise(invalid: object) -> None:
    frame = pd.DataFrame({"FTR": [invalid], "FTHome": [1], "FTAway": [0]})

    with pytest.raises(LabelValidationError, match="outside H/D/A"):
        add_target_labels(frame)


def test_label_builder_does_not_mutate_source() -> None:
    frame = pd.DataFrame({"FTR": ["D"], "FTHome": [1], "FTAway": [1]})
    before = frame.copy(deep=True)

    add_target_labels(frame)

    pd.testing.assert_frame_equal(frame, before)
