"""Typed, explicit-allowlist catalogue for the closed public research programme."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

PROGRAMME_ID = "football_outcome_lab_public_v1"
PROGRAMME_CONCLUSION = "DATA_CEILING_UPHELD"
DEFAULT_MODEL = "LR52"


class CatalogValidationError(ValueError):
    """Raised when a committed research record violates the catalogue schema."""


class CrossPopulationComparisonError(ValueError):
    """Raised when raw scores from different populations are ranked together."""


class ExperimentRole(StrEnum):
    """The three programme roles represented by committed records."""

    BASELINE = "baseline"
    EXTERNAL_BENCHMARK = "external_benchmark"
    RESEARCH_CANDIDATE = "research_candidate"


@dataclass(frozen=True, slots=True)
class ResultSpec:
    """One explicit committed record and its normalization adapter."""

    record_id: str
    relative_file: str
    role: ExperimentRole
    adapter: str
    display_name: str
    population_id: str
    population_description: str
    coverage_note: str
    timing_limitation: str


RESULT_ALLOWLIST: tuple[ResultSpec, ...] = (
    ResultSpec(
        record_id="lr52_baseline",
        relative_file="research/lr52_baseline/result.json",
        role=ExperimentRole.BASELINE,
        adapter="lr52_baseline",
        display_name="LR52",
        population_id="nine_league_canonical_22617",
        population_description="Nine-league canonical historical population",
        coverage_note="Complete 22,617-row canonical population.",
        timing_limitation="Only the fixed 52 approved pre-match features are used.",
    ),
    ResultSpec(
        record_id="closing_market_benchmark",
        relative_file="research/closing_market_benchmark/result.json",
        role=ExperimentRole.EXTERNAL_BENCHMARK,
        adapter="closing_market",
        display_name="Closing market",
        population_id="nine_league_canonical_22617",
        population_description="Nine-league canonical rows with complete closing prices",
        coverage_note="Exact-key coverage is 100% for the committed 22,617-row market table.",
        timing_limitation=(
            "Closing prices are a near-kickoff external benchmark, not an internal candidate."
        ),
    ),
    ResultSpec(
        record_id="match2vec",
        relative_file="research/match2vec/result.json",
        role=ExperimentRole.RESEARCH_CANDIDATE,
        adapter="standard_candidate",
        display_name="Match2Vec",
        population_id="nine_league_canonical_22617",
        population_description="Nine-league canonical historical population",
        coverage_note="Uses the full canonical LR52 comparison population.",
        timing_limitation=(
            "Sequences and descriptors use earlier matches; 2025 has already been "
            "examined historically."
        ),
    ),
    ResultSpec(
        record_id="pseudo_xg",
        relative_file="research/pseudo_xg/result.json",
        role=ExperimentRole.RESEARCH_CANDIDATE,
        adapter="standard_candidate",
        display_name="Pseudo-xG",
        population_id="nine_league_canonical_22617",
        population_description="Nine-league canonical historical population",
        coverage_note="Uses the full canonical LR52 comparison population.",
        timing_limitation=(
            "Pseudo-xG is estimated from completed prior match statistics, not "
            "same-match shot quality."
        ),
    ),
    ResultSpec(
        record_id="understat_xg",
        relative_file="research/understat_xg/result.json",
        role=ExperimentRole.RESEARCH_CANDIDATE,
        adapter="standard_candidate",
        display_name="Understat xG",
        population_id="big5_external_xg_partial_8953",
        population_description=(
            "Big-5 canonical population with partial historical external-xG coverage"
        ),
        coverage_note=(
            "8,953 aligned historical rows; external xG ends after season start 2023 "
            "and later folds lack within-season xG history."
        ),
        timing_limitation="Final post-match xG is used only as lagged history for later fixtures.",
    ),
    ResultSpec(
        record_id="deep_capacity",
        relative_file="research/deep_capacity/result.json",
        role=ExperimentRole.RESEARCH_CANDIDATE,
        adapter="deep_capacity",
        display_name="Deep MLP",
        population_id="nine_league_canonical_22617",
        population_description="Nine-league canonical historical population",
        coverage_note="Uses the full canonical LR52 comparison population.",
        timing_limitation=(
            "The complete-date public validation split changed the selected "
            "architecture; 2025 is historical, not currently unseen."
        ),
    ),
    ResultSpec(
        record_id="graph_model",
        relative_file="research/graph_model/result.json",
        role=ExperimentRole.RESEARCH_CANDIDATE,
        adapter="graph_model",
        display_name="Graph SVD8",
        population_id="big5_transfermarkt_exact_12452",
        population_description="Exact-key Big-5 Transfermarkt-covered population",
        coverage_note=(
            "12,452 matched rows; development contains 5,328 identical candidate/LR52 rows."
        ),
        timing_limitation="Graph relations and representations use completed earlier dates only.",
    ),
    ResultSpec(
        record_id="player_availability",
        relative_file="research/player_availability/result.json",
        role=ExperimentRole.RESEARCH_CANDIDATE,
        adapter="standard_candidate",
        display_name="Player availability",
        population_id="big5_transfermarkt_exact_12452",
        population_description="Exact-key Big-5 Transfermarkt-covered population",
        coverage_note=(
            "12,452 of 12,458 canonical Big-5 rows; filtering changes the aligned LR52 population."
        ),
        timing_limitation=(
            "Target lineup and bench timestamps are unknown and excluded; only "
            "completed-prior history is used."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """One immutable probabilistic metric aggregate."""

    n_samples: int
    log_loss: float
    brier_score: float
    ece: float

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise CatalogValidationError("metric aggregate requires positive n_samples")
        if not all(math.isfinite(value) for value in (self.log_loss, self.brier_score, self.ece)):
            raise CatalogValidationError("metric aggregate contains a non-finite value")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_samples": self.n_samples,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "ece": self.ece,
        }


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    """The stable default model record."""

    experiment_id: str
    name: str
    role: ExperimentRole
    population_id: str
    population_description: str
    development: MetricSummary
    historical_final: MetricSummary
    selected_as_default: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "role": self.role.value,
            "population_id": self.population_id,
            "population_description": self.population_description,
            "development": self.development.to_dict(),
            "historical_final": self.historical_final.to_dict(),
            "selected_as_default": self.selected_as_default,
        }


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkSummary:
    """A non-candidate external benchmark on an aligned population."""

    experiment_id: str
    name: str
    role: ExperimentRole
    population_id: str
    population_description: str
    development_samples: int
    aligned_lr52_log_loss: float
    benchmark_log_loss: float
    benchmark_advantage_log_loss: float
    historical_final_advantage_log_loss: float
    gate_applicable: bool
    coverage_note: str
    timing_limitation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "role": self.role.value,
            "population_id": self.population_id,
            "population_description": self.population_description,
            "development_samples": self.development_samples,
            "aligned_lr52_log_loss": self.aligned_lr52_log_loss,
            "benchmark_log_loss": self.benchmark_log_loss,
            "benchmark_advantage_log_loss": self.benchmark_advantage_log_loss,
            "historical_final_advantage_log_loss": self.historical_final_advantage_log_loss,
            "gate_applicable": self.gate_applicable,
            "coverage_note": self.coverage_note,
            "timing_limitation": self.timing_limitation,
        }


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """The common predefined candidate development gate."""

    minimum_improvement_log_loss: float
    minimum_improving_folds: int
    development_fold_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "primary_metric": "development_sample_count_weighted_log_loss",
            "improvement_sign_convention": "aligned_lr52_log_loss_minus_candidate_log_loss",
            "minimum_improvement_log_loss": self.minimum_improvement_log_loss,
            "minimum_improving_folds": self.minimum_improving_folds,
            "development_fold_count": self.development_fold_count,
        }


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """One within-population candidate comparison normalized from its authority."""

    experiment_id: str
    name: str
    role: ExperimentRole
    population_id: str
    population_description: str
    development_samples: int
    aligned_lr52_log_loss: float
    candidate_log_loss: float
    candidate_improvement_log_loss: float
    improving_development_folds: int
    gate_policy: GatePolicy
    gate_passed: bool
    verdict: str
    disposition: str
    historical_final_improvement_log_loss: float
    coverage_note: str
    timing_limitation: str
    selected_as_default: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "role": self.role.value,
            "population_id": self.population_id,
            "population_description": self.population_description,
            "development_samples": self.development_samples,
            "aligned_lr52_log_loss": self.aligned_lr52_log_loss,
            "candidate_log_loss": self.candidate_log_loss,
            "candidate_improvement_log_loss": self.candidate_improvement_log_loss,
            "improving_development_folds": self.improving_development_folds,
            "required_gate": self.gate_policy.to_dict(),
            "gate_passed": self.gate_passed,
            "verdict": self.verdict,
            "disposition": self.disposition,
            "historical_final_improvement_log_loss": self.historical_final_improvement_log_loss,
            "coverage_note": self.coverage_note,
            "timing_limitation": self.timing_limitation,
            "selected_as_default": self.selected_as_default,
        }


@dataclass(frozen=True, slots=True)
class RecordHash:
    """A logical protected-record identifier and its byte hash."""

    record_id: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"record_id": self.record_id, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SourceInventoryEntry:
    """One discovered completed source experiment or supporting audit."""

    experiment_name: str
    executable_authority: str
    structured_result: str
    public_representation: str
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "experiment_name": self.experiment_name,
            "executable_authority": self.executable_authority,
            "structured_result": self.structured_result,
            "public_representation": self.public_representation,
            "exclusion_reason": self.exclusion_reason,
        }


SOURCE_INVENTORY: tuple[SourceInventoryEntry, ...] = (
    SourceInventoryEntry(
        "LR52 baseline", "new baseline runner", "new baseline record", "complete", None
    ),
    SourceInventoryEntry(
        "Match2Vec sequence audit",
        "new Match2Vec runner",
        "new Match2Vec development and final records",
        "complete",
        None,
    ),
    SourceInventoryEntry(
        "Pseudo-xG audit", "new xG runner", "new xG evaluation record", "complete", None
    ),
    SourceInventoryEntry(
        "Real historical xG audit",
        "new real-xG runner",
        "new real-xG evaluation record",
        "complete as Understat xG",
        None,
    ),
    SourceInventoryEntry(
        "Deep model audit",
        "new deep audit runner",
        "new deep audit record",
        "internal capacity context represented",
        "Market and timing-uncertain availability neural contexts remain outside "
        "the internal-candidate catalogue.",
    ),
    SourceInventoryEntry(
        "Temporal graph embedding audit",
        "new graph audit runner",
        "new graph evaluation record",
        "internal LR52 plus graph context represented",
        "Market and combined availability contexts are outside the Phase 5C2 contract.",
    ),
    SourceInventoryEntry(
        "Player-availability audit",
        "new availability runner",
        "new availability evaluation record",
        "strictly-prior context represented",
        "Target official lineup and bench context lacks source publication "
        "timestamps and is excluded.",
    ),
    SourceInventoryEntry(
        "Market-aware combination audit",
        "new market-aware runner",
        "new market-aware record",
        "closing benchmark represented",
        "Fitted odds calibration, blends, and market-plus-internal candidates "
        "require a separate public market-derived synthesis phase.",
    ),
    SourceInventoryEntry(
        "Odds-movement audit",
        "new odds-movement runner",
        "new odds-movement record",
        "closing benchmark represented",
        "Movement and market-plus-internal candidates are outside the six internal "
        "advanced candidates.",
    ),
    SourceInventoryEntry(
        "Odds-derived feature and stacking audit",
        "new odds-feature runner",
        "new odds-feature record",
        "not represented",
        "Completed market-derived candidate family is proposed for a later "
        "source-verification phase.",
    ),
    SourceInventoryEntry(
        "Open candidate discovery audit",
        "new open-candidate runner",
        "new open-candidate record",
        "not represented",
        "Completed recalibration and stacking families are proposed for a later "
        "source-verification phase.",
    ),
    SourceInventoryEntry(
        "Match2Vec robustness and final audit",
        "new robustness and final runners",
        "new robustness and final records",
        "supporting evidence under Match2Vec",
        "Supporting stages are not separate candidate families.",
    ),
    SourceInventoryEntry(
        "Legacy model-family comparison",
        "legacy family-comparison runner",
        "legacy family-comparison metrics",
        "not represented",
        "Legacy split and schema differ from the stable public protocol; source "
        "verification is proposed later.",
    ),
    SourceInventoryEntry(
        "Legacy model exploration and tuning",
        "legacy exploration runner",
        "legacy exploration metrics",
        "not represented",
        "Legacy validation and test semantics require a separate audit before "
        "public normalization.",
    ),
    SourceInventoryEntry(
        "Legacy static style clustering",
        "legacy clustering runner",
        "legacy clustering metrics",
        "not represented",
        "Legacy fold and feature contracts require a separate source-verification phase.",
    ),
    SourceInventoryEntry(
        "Legacy opponent-style interactions",
        "legacy interaction runner",
        "legacy interaction metrics",
        "not represented",
        "Legacy fold and feature contracts require a separate source-verification phase.",
    ),
    SourceInventoryEntry(
        "Legacy walk-forward robustness",
        "legacy walk-forward runner",
        "legacy walk-forward metrics",
        "not represented",
        "This is a protocol robustness study rather than one of the six normalized "
        "advanced candidates.",
    ),
)


PROGRAMME_LIMITATIONS: tuple[str, ...] = (
    "The evidence is observational historical evaluation, not a causal experiment.",
    "Candidate populations and coverage differ, so raw Log Loss is not ranked across experiments.",
    "Post-match participation and xG variables are used only as lagged completed-match history.",
    "Broad, timestamp-verified pre-kickoff availability and lineup data is incomplete.",
    "The 2025 historical-final folds have already been examined; there is no "
    "current unseen final set.",
    "The conclusion does not prove LR52 is globally optimal or that no future model can improve.",
    "The conclusion does not prove the market cannot be approached.",
    "No causal, profitability, operational-utility, live-data, or deployment claim is made.",
)

NEXT_RESEARCH_PRIORITIES: tuple[str, ...] = (
    "Acquire broad, timestamp-verified pre-kickoff availability, lineup, injury, "
    "and suspension data.",
    "Add timestamped transfer and manager-change context with explicit effective dates.",
    "Evaluate richer event and shot-location data with broad season and division coverage.",
    "Create a new untouched temporal final set before any future candidate selection.",
    "Source-verify the completed market-derived and legacy tree/interaction audits "
    "in later phases rather than mixing their raw scores into v1.",
)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(f"{context} must be an object")
    return cast(Mapping[str, Any], value)


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise CatalogValidationError(f"{context} is missing required field {key}")
    return mapping[key]


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value:
        raise CatalogValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _number(mapping: Mapping[str, Any], key: str, context: str) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogValidationError(f"{context}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CatalogValidationError(f"{context}.{key} must be finite")
    return result


def _integer(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogValidationError(f"{context}.{key} must be an integer")
    return cast(int, value)


def _boolean(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = _required(mapping, key, context)
    if not isinstance(value, bool):
        raise CatalogValidationError(f"{context}.{key} must be boolean")
    return value


def _metric(value: object, context: str) -> MetricSummary:
    mapping = _mapping(value, context)
    return MetricSummary(
        n_samples=_integer(mapping, "n_samples", context),
        log_loss=_number(mapping, "log_loss", context),
        brier_score=_number(mapping, "brier_score", context),
        ece=_number(mapping, "ece", context),
    )


def _metric_with_parent_samples(
    value: object,
    context: str,
    parent: Mapping[str, Any],
) -> MetricSummary:
    mapping = _mapping(value, context)
    if "n_samples" in mapping:
        return _metric(mapping, context)
    return MetricSummary(
        n_samples=_integer(parent, "n_samples", context.rsplit(".", 1)[0]),
        log_loss=_number(mapping, "log_loss", context),
        brier_score=_number(mapping, "brier_score", context),
        ece=_number(mapping, "ece", context),
    )


def _find_candidate(
    values: object,
    *,
    context: str,
    key_values: Mapping[str, str],
) -> Mapping[str, Any]:
    if not isinstance(values, list):
        raise CatalogValidationError(f"{context} must be a list")
    found = [
        _mapping(value, context)
        for value in values
        if all(
            _mapping(value, context).get(key) == expected for key, expected in key_values.items()
        )
    ]
    if len(found) != 1:
        raise CatalogValidationError(f"{context} must contain exactly one selected candidate")
    return found[0]


def _candidate_metrics(
    record: Mapping[str, Any],
    spec: ResultSpec,
) -> tuple[MetricSummary, MetricSummary, int, float]:
    if spec.adapter == "standard_candidate":
        development = _mapping(_required(record, "development", spec.record_id), "development")
        final = _mapping(_required(record, "historical_final", spec.record_id), "historical_final")
        if "improving_fold_count" in development:
            improving_folds = _integer(development, "improving_fold_count", "development")
        else:
            gate = _mapping(_required(record, "success_gate", spec.record_id), "success_gate")
            observed = _mapping(
                _required(gate, "observed", "success_gate"), "success_gate.observed"
            )
            improving_folds = _integer(observed, "improving_fold_count", "success_gate.observed")
        return (
            _metric_with_parent_samples(
                _required(development, "lr52", "development"),
                "development.lr52",
                development,
            ),
            _metric_with_parent_samples(
                _required(development, "candidate", "development"),
                "development.candidate",
                development,
            ),
            improving_folds,
            _number(final, "candidate_improvement_log_loss", "historical_final"),
        )
    if spec.adapter == "deep_capacity":
        baseline = _mapping(_required(record, "baseline_lr52", spec.record_id), "baseline_lr52")
        selection = _mapping(_required(record, "selection", spec.record_id), "selection")
        architecture = _text(selection, "safety_adapted_selected_architecture", "selection")
        selected = _find_candidate(
            _required(record, "development_candidates", spec.record_id),
            context="development_candidates",
            key_values={"architecture": architecture},
        )
        development = _mapping(
            _required(selected, "development", "selected candidate"), "selected development"
        )
        final = _mapping(_required(record, "historical_final", spec.record_id), "historical_final")
        return (
            _metric(
                _required(baseline, "development", "baseline_lr52"), "baseline_lr52.development"
            ),
            _metric(development, "selected development"),
            _integer(development, "improving_fold_count", "selected development"),
            _number(final, "candidate_improvement_log_loss", "historical_final"),
        )
    if spec.adapter == "graph_model":
        baseline = _mapping(_required(record, "baseline_lr52", spec.record_id), "baseline_lr52")
        selection = _mapping(_required(record, "selection", spec.record_id), "selection")
        selected = _find_candidate(
            _required(record, "development_candidates", spec.record_id),
            context="development_candidates",
            key_values={
                "configuration": _text(selection, "selected_configuration", "selection"),
                "context": _text(selection, "selected_context", "selection"),
            },
        )
        development = _mapping(
            _required(selected, "development", "selected candidate"), "selected development"
        )
        gate = _mapping(_required(record, "success_gate", spec.record_id), "success_gate")
        observed = _mapping(_required(gate, "observed", "success_gate"), "success_gate.observed")
        return (
            _metric(
                _required(baseline, "development", "baseline_lr52"), "baseline_lr52.development"
            ),
            _metric(development, "selected development"),
            _integer(development, "improving_fold_count", "selected development"),
            _number(observed, "historical_final_improvement_log_loss", "success_gate.observed"),
        )
    raise CatalogValidationError(f"unsupported candidate adapter {spec.adapter}")


def _gate_policy(record: Mapping[str, Any], record_id: str) -> tuple[GatePolicy, Mapping[str, Any]]:
    gate = _mapping(_required(record, "success_gate", record_id), "success_gate")
    definition = _mapping(_required(gate, "definition", "success_gate"), "success_gate.definition")
    return (
        GatePolicy(
            minimum_improvement_log_loss=_number(
                definition, "minimum_improvement_log_loss", "success_gate.definition"
            ),
            minimum_improving_folds=_integer(
                definition, "minimum_improving_folds", "success_gate.definition"
            ),
            development_fold_count=_integer(
                definition, "development_fold_count", "success_gate.definition"
            ),
        ),
        gate,
    )


def _normalize_candidate(record: Mapping[str, Any], spec: ResultSpec) -> CandidateSummary:
    lr52, candidate, improving_folds, final_improvement = _candidate_metrics(record, spec)
    if lr52.n_samples != candidate.n_samples:
        raise CatalogValidationError(f"{spec.record_id} candidate and LR52 sample counts differ")
    improvement = lr52.log_loss - candidate.log_loss
    if spec.adapter in {"deep_capacity", "graph_model"}:
        stored_improvement = _number(
            _mapping(
                _required(
                    _find_candidate(
                        _required(record, "development_candidates", spec.record_id),
                        context="development_candidates",
                        key_values=(
                            {
                                "architecture": _text(
                                    _mapping(
                                        _required(record, "selection", spec.record_id), "selection"
                                    ),
                                    "safety_adapted_selected_architecture",
                                    "selection",
                                )
                            }
                            if spec.adapter == "deep_capacity"
                            else {
                                "configuration": _text(
                                    _mapping(
                                        _required(record, "selection", spec.record_id), "selection"
                                    ),
                                    "selected_configuration",
                                    "selection",
                                ),
                                "context": _text(
                                    _mapping(
                                        _required(record, "selection", spec.record_id), "selection"
                                    ),
                                    "selected_context",
                                    "selection",
                                ),
                            }
                        ),
                    ),
                    "development",
                    "selected candidate",
                ),
                "selected development",
            ),
            "candidate_improvement_log_loss",
            "selected development",
        )
    else:
        development = _mapping(_required(record, "development", spec.record_id), "development")
        stored_improvement = _number(development, "candidate_improvement_log_loss", "development")
    if not math.isclose(improvement, stored_improvement, rel_tol=0.0, abs_tol=1e-12):
        raise CatalogValidationError(
            f"{spec.record_id} stored improvement has the wrong sign or value"
        )

    policy, gate = _gate_policy(record, spec.record_id)
    observed = _mapping(_required(gate, "observed", "success_gate"), "success_gate.observed")
    if not math.isclose(
        _number(observed, "candidate_improvement_log_loss", "success_gate.observed"),
        improvement,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CatalogValidationError(f"{spec.record_id} gate improvement differs from metrics")
    if _integer(observed, "improving_fold_count", "success_gate.observed") != improving_folds:
        raise CatalogValidationError(f"{spec.record_id} gate fold count differs from metrics")
    expected_pass = (
        improvement >= policy.minimum_improvement_log_loss
        and improving_folds >= policy.minimum_improving_folds
    )
    gate_passed = _boolean(gate, "passed", "success_gate")
    if gate_passed != expected_pass:
        raise CatalogValidationError(f"{spec.record_id} gate verdict does not match its thresholds")
    return CandidateSummary(
        experiment_id=_text(record, "experiment_id", spec.record_id),
        name=spec.display_name,
        role=spec.role,
        population_id=spec.population_id,
        population_description=spec.population_description,
        development_samples=candidate.n_samples,
        aligned_lr52_log_loss=lr52.log_loss,
        candidate_log_loss=candidate.log_loss,
        candidate_improvement_log_loss=improvement,
        improving_development_folds=improving_folds,
        gate_policy=policy,
        gate_passed=gate_passed,
        verdict=_text(record, "verdict", spec.record_id),
        disposition=_text(record, "disposition", spec.record_id),
        historical_final_improvement_log_loss=final_improvement,
        coverage_note=spec.coverage_note,
        timing_limitation=spec.timing_limitation,
        selected_as_default=False,
    )


@dataclass(frozen=True, slots=True)
class ResearchCatalog:
    """Validated immutable catalogue and synthesis metadata."""

    schema_version: str
    programme_id: str
    status: str
    default_model: str
    conclusion: str
    baseline: BaselineSummary
    external_benchmark: ExternalBenchmarkSummary
    candidates: tuple[CandidateSummary, ...]
    gate_policy: GatePolicy
    candidate_ranking_by_valid_development_improvement: tuple[str, ...]
    source_inventory: tuple[SourceInventoryEntry, ...]
    protected_record_hashes: tuple[RecordHash, ...]

    def __post_init__(self) -> None:
        if self.default_model != self.baseline.name or not self.baseline.selected_as_default:
            raise CatalogValidationError("LR52 must remain the selected default")
        if any(candidate.selected_as_default for candidate in self.candidates):
            raise CatalogValidationError("a rejected research candidate cannot be the default")
        if any(candidate.gate_passed for candidate in self.candidates):
            raise CatalogValidationError("DATA_CEILING_UPHELD requires every advanced gate to fail")
        if self.conclusion != PROGRAMME_CONCLUSION:
            raise CatalogValidationError("programme conclusion is not the predefined closure")

    def rank_candidates_by_raw_log_loss(self) -> tuple[str, ...]:
        """Rank only when all candidates share one population; otherwise fail."""

        populations = {candidate.population_id for candidate in self.candidates}
        if len(populations) != 1:
            raise CrossPopulationComparisonError(
                "raw candidate Log Loss cannot be ranked across coverage populations"
            )
        return tuple(
            candidate.experiment_id
            for candidate in sorted(
                self.candidates,
                key=lambda item: (item.candidate_log_loss, item.experiment_id),
            )
        )

    def to_summary_dict(self) -> dict[str, Any]:
        ranking = []
        by_id = {candidate.experiment_id: candidate for candidate in self.candidates}
        for index, experiment_id in enumerate(
            self.candidate_ranking_by_valid_development_improvement, start=1
        ):
            candidate = by_id[experiment_id]
            ranking.append(
                {
                    "rank": index,
                    "experiment_id": experiment_id,
                    "name": candidate.name,
                    "candidate_improvement_log_loss": (candidate.candidate_improvement_log_loss),
                    "population_id": candidate.population_id,
                    "raw_score_ranked_across_populations": False,
                }
            )
        return {
            "schema_version": self.schema_version,
            "programme_id": self.programme_id,
            "status": self.status,
            "default_model": {
                "name": self.default_model,
                "experiment_id": self.baseline.experiment_id,
                "selected": True,
                "reason": "No tested advanced candidate passed its predefined development gate.",
            },
            "conclusion": self.conclusion,
            "baseline": self.baseline.to_dict(),
            "external_benchmark": self.external_benchmark.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "gate_policy": self.gate_policy.to_dict(),
            "candidate_ranking_by_valid_development_improvement": ranking,
            "coverage_notes": [
                "Raw candidate Log Loss is never ranked across different coverage populations.",
                "Every improvement is computed against LR52 on that candidate's own aligned rows.",
                "Population identifiers make differing evaluation scopes explicit.",
            ],
            "limitations": list(PROGRAMME_LIMITATIONS),
            "next_research_priorities": list(NEXT_RESEARCH_PRIORITIES),
            "source_inventory": [item.to_dict() for item in self.source_inventory],
            "protected_record_hashes": [item.to_dict() for item in self.protected_record_hashes],
        }


def _validate_record_header(record: Mapping[str, Any], spec: ResultSpec) -> str:
    _text(record, "schema_version", spec.record_id)
    status = _text(record, "status", spec.record_id)
    if status != "reproduced":
        raise CatalogValidationError(f"{spec.record_id} is not a reproduced authority")
    return _text(record, "experiment_id", spec.record_id)


def _normalize_baseline(record: Mapping[str, Any], spec: ResultSpec) -> BaselineSummary:
    development = _mapping(_required(record, "development", spec.record_id), "development")
    final = _mapping(_required(record, "historical_final", spec.record_id), "historical_final")
    return BaselineSummary(
        experiment_id=_text(record, "experiment_id", spec.record_id),
        name=spec.display_name,
        role=spec.role,
        population_id=spec.population_id,
        population_description=spec.population_description,
        development=_metric(_required(development, "model", "development"), "development.model"),
        historical_final=_metric(
            _required(final, "model", "historical_final"),
            "historical_final.model",
        ),
        selected_as_default=True,
    )


def _normalize_external(record: Mapping[str, Any], spec: ResultSpec) -> ExternalBenchmarkSummary:
    development = _mapping(_required(record, "development", spec.record_id), "development")
    final = _mapping(_required(record, "historical_final", spec.record_id), "historical_final")
    lr52 = _metric(
        _required(development, "aligned_lr52", "development"), "development.aligned_lr52"
    )
    market = _metric(_required(development, "market", "development"), "development.market")
    final_lr52 = _metric(
        _required(final, "aligned_lr52", "historical_final"),
        "historical_final.aligned_lr52",
    )
    final_market = _metric(
        _required(final, "market", "historical_final"), "historical_final.market"
    )
    if lr52.n_samples != market.n_samples:
        raise CatalogValidationError("closing market development rows are not aligned")
    if final_lr52.n_samples != final_market.n_samples:
        raise CatalogValidationError("closing market historical-final rows are not aligned")
    return ExternalBenchmarkSummary(
        experiment_id=_text(record, "experiment_id", spec.record_id),
        name=spec.display_name,
        role=spec.role,
        population_id=spec.population_id,
        population_description=spec.population_description,
        development_samples=market.n_samples,
        aligned_lr52_log_loss=lr52.log_loss,
        benchmark_log_loss=market.log_loss,
        benchmark_advantage_log_loss=lr52.log_loss - market.log_loss,
        historical_final_advantage_log_loss=final_lr52.log_loss - final_market.log_loss,
        gate_applicable=False,
        coverage_note=spec.coverage_note,
        timing_limitation=spec.timing_limitation,
    )


def build_research_catalog(
    records: Mapping[str, Mapping[str, Any]],
    *,
    protected_hashes: Mapping[str, str] | None = None,
) -> ResearchCatalog:
    """Validate and normalize exactly the eight allowlisted committed records."""

    expected_ids = tuple(spec.record_id for spec in RESULT_ALLOWLIST)
    missing = [record_id for record_id in expected_ids if record_id not in records]
    unexpected = sorted(set(records).difference(expected_ids))
    if missing or unexpected:
        raise CatalogValidationError(
            "catalogue record set differs from allowlist; "
            f"missing={missing}, unexpected={unexpected}"
        )
    experiment_ids = [
        _validate_record_header(records[spec.record_id], spec) for spec in RESULT_ALLOWLIST
    ]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise CatalogValidationError("catalogue contains duplicate experiment IDs")

    baseline_spec = next(spec for spec in RESULT_ALLOWLIST if spec.role is ExperimentRole.BASELINE)
    external_spec = next(
        spec for spec in RESULT_ALLOWLIST if spec.role is ExperimentRole.EXTERNAL_BENCHMARK
    )
    baseline = _normalize_baseline(records[baseline_spec.record_id], baseline_spec)
    external = _normalize_external(records[external_spec.record_id], external_spec)
    candidates = tuple(
        _normalize_candidate(records[spec.record_id], spec)
        for spec in RESULT_ALLOWLIST
        if spec.role is ExperimentRole.RESEARCH_CANDIDATE
    )
    policies = {candidate.gate_policy for candidate in candidates}
    if len(policies) != 1:
        raise CatalogValidationError("advanced candidates do not share the predefined gate")
    policy = next(iter(policies))
    ranking = tuple(
        candidate.experiment_id
        for candidate in sorted(
            candidates,
            key=lambda item: (-item.candidate_improvement_log_loss, item.experiment_id),
        )
    )
    hashes = protected_hashes or {}
    protected = tuple(
        RecordHash(record_id=record_id, sha256=hashes[record_id])
        for record_id in expected_ids
        if record_id in hashes
    )
    return ResearchCatalog(
        schema_version="1.0",
        programme_id=PROGRAMME_ID,
        status="closed_research_programme",
        default_model=DEFAULT_MODEL,
        conclusion=PROGRAMME_CONCLUSION,
        baseline=baseline,
        external_benchmark=external,
        candidates=candidates,
        gate_policy=policy,
        candidate_ranking_by_valid_development_improvement=ranking,
        source_inventory=SOURCE_INVENTORY,
        protected_record_hashes=protected,
    )


def load_research_catalog(repository_root: Path) -> ResearchCatalog:
    """Read only the explicit allowlist; perform no discovery or network access."""

    records: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}
    for spec in RESULT_ALLOWLIST:
        path = repository_root / Path(spec.relative_file)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise CatalogValidationError(
                f"cannot read allowlisted record {spec.record_id}"
            ) from exc
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogValidationError(
                f"allowlisted record {spec.record_id} is invalid JSON"
            ) from exc
        records[spec.record_id] = _mapping(value, spec.record_id)
        hashes[spec.record_id] = hashlib.sha256(payload).hexdigest()
    return build_research_catalog(records, protected_hashes=hashes)


def rank_same_population_by_raw_log_loss(
    candidates: Sequence[CandidateSummary],
) -> tuple[str, ...]:
    """Expose the population guard for a caller-provided candidate subset."""

    populations = {candidate.population_id for candidate in candidates}
    if len(populations) > 1:
        raise CrossPopulationComparisonError(
            "raw candidate Log Loss cannot be ranked across coverage populations"
        )
    return tuple(
        candidate.experiment_id
        for candidate in sorted(
            candidates,
            key=lambda item: (item.candidate_log_loss, item.experiment_id),
        )
    )
