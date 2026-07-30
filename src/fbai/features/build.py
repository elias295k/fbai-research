"""Public orchestration and verified Parquet output for Phase 2B features."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fbai.data.schema import CANONICAL_COLUMNS, sort_canonical_frame
from fbai.data.store import discover_canonical_files
from fbai.features.checks import validate_feature_table
from fbai.features.context import add_context_features
from fbai.features.elo import add_elo_features
from fbai.features.labels import add_target_labels
from fbai.features.rolling import add_rolling_features
from fbai.features.schema import FEATURE_TABLE_COLUMNS

_SAFE_PARTITION = re.compile(r"^[A-Za-z0-9_-]+$")


class FeatureWriteError(RuntimeError):
    """Raised when feature partitions cannot be written and verified."""


@dataclass(frozen=True)
class FeaturePartitionWrite:
    """One verified per-division feature partition."""

    division: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class FeatureWriteResult:
    """Structured summary of a feature partition write."""

    input_row_count: int
    output_row_count: int
    partitions: tuple[FeaturePartitionWrite, ...]


def build_feature_table(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build the validated, stably ordered Phase 2B feature table."""

    input_row_count = len(canonical)
    ordered = sort_canonical_frame(canonical)
    featured = add_target_labels(ordered)
    featured = add_elo_features(featured)
    featured = add_context_features(featured)
    featured = add_rolling_features(featured)
    result = featured.loc[:, list(FEATURE_TABLE_COLUMNS)].reset_index(drop=True)
    validate_feature_table(result, expected_row_count=input_row_count)
    return result


def build_feature_table_from_parquet(directory: Path) -> pd.DataFrame:
    """Load deterministic canonical partitions and build their feature table."""

    files = discover_canonical_files(Path(directory))
    canonical = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in files],
        ignore_index=True,
    )
    canonical = canonical.loc[:, list(CANONICAL_COLUMNS)]
    return build_feature_table(canonical)


def _verify_round_trip(expected: pd.DataFrame, temporary_path: Path) -> None:
    actual = pd.read_parquet(temporary_path, engine="pyarrow")
    actual = actual.loc[:, list(FEATURE_TABLE_COLUMNS)]
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_like=False)
    except AssertionError as exc:
        raise FeatureWriteError(
            f"Parquet round-trip verification failed for {temporary_path.name}: {exc}"
        ) from exc
    validate_feature_table(actual, expected_row_count=len(expected))


def write_feature_partitions(
    features: pd.DataFrame,
    destination: Path,
) -> FeatureWriteResult:
    """Write one validated and read-back-verified Parquet file per division."""

    validate_feature_table(features)
    output_directory = Path(destination)
    divisions = sorted(features["Division"].unique().tolist())
    for division in divisions:
        if not _SAFE_PARTITION.fullmatch(str(division)):
            raise FeatureWriteError(f"Unsafe division partition name: {division!r}")

    output_directory.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, pd.DataFrame, Path, Path]] = []
    temporary_paths: list[Path] = []
    try:
        for division in divisions:
            partition = (
                features.loc[
                    features["Division"].eq(division),
                    list(FEATURE_TABLE_COLUMNS),
                ]
                .copy()
                .reset_index(drop=True)
            )
            validate_feature_table(partition, expected_row_count=len(partition))
            final_path = output_directory / f"{division}.parquet"
            with tempfile.NamedTemporaryFile(
                prefix=f".{division}.",
                suffix=".parquet.tmp",
                dir=output_directory,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            temporary_paths.append(temporary_path)
            partition.to_parquet(
                temporary_path,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            _verify_round_trip(partition, temporary_path)
            pending.append((str(division), partition, temporary_path, final_path))

        for _division, _partition, temporary_path, final_path in pending:
            os.replace(temporary_path, final_path)
            temporary_paths.remove(temporary_path)

        partitions = tuple(
            FeaturePartitionWrite(
                division=division,
                path=final_path,
                row_count=len(partition),
            )
            for division, partition, _temporary_path, final_path in pending
        )
        return FeatureWriteResult(
            input_row_count=len(features),
            output_row_count=sum(partition.row_count for partition in partitions),
            partitions=partitions,
        )
    except Exception:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise
