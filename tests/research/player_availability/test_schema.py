from __future__ import annotations

import hashlib

from fbai.features import FEATURE_COLUMNS
from fbai.research.player_availability.schema import (
    AVAILABILITY_CLASSIFIER_CONFIG,
    PRIOR_AVAILABILITY_FEATURES,
    PRIOR_BASE_FEATURES,
    TIMING_AUDIT,
    InformationTiming,
)


def test_exact_authoritative_feature_contract() -> None:
    assert len(PRIOR_BASE_FEATURES) == 21
    assert len(PRIOR_AVAILABILITY_FEATURES) == 63
    assert len(set(PRIOR_AVAILABILITY_FEATURES)) == 63
    assert all(feature.endswith("_pre") for feature in PRIOR_AVAILABILITY_FEATURES)
    assert not any("near_kickoff" in feature for feature in PRIOR_AVAILABILITY_FEATURES)


def test_timing_audit_forbids_unproven_target_information() -> None:
    timing = {item.source_group: item for item in TIMING_AUDIT}

    assert timing["appearances"].timing is InformationTiming.COMPLETED_PRIOR_MATCH_ONLY
    assert timing["target_official_lineup_and_bench"].timing is InformationTiming.TIMING_UNKNOWN
    assert timing["target_official_lineup_and_bench"].public_target_use == "forbidden"
    assert (
        timing["target_participation_and_substitutions"].timing
        is InformationTiming.KNOWN_ONLY_AFTER_TARGET_MATCH
    )
    assert timing["injury_suspension_absence_labels"].public_target_use == "not_available"


def test_source_classifier_and_stable_fingerprint_are_exact() -> None:
    assert AVAILABILITY_CLASSIFIER_CONFIG.input_feature_count == 115
    assert AVAILABILITY_CLASSIFIER_CONFIG.parameter_count == 348
    assert AVAILABILITY_CLASSIFIER_CONFIG.identifier == (
        "lr115_median_standardized_lbfgs_l2_c1_iter2000_seed42"
    )
    fingerprint = hashlib.sha256("\0".join(FEATURE_COLUMNS).encode()).hexdigest()
    assert fingerprint == "09b9204c283e8adf7e91e98cb1a547183b3f832cc95f815b9845208be822de78"
