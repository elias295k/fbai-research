"""Validated, partitioned Parquet output for canonical match data."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fbai.data.audit import audit_canonical
from fbai.data.schema import CANONICAL_COLUMNS, sort_canonical_frame

_SAFE_PARTITION = re.compile(r"^[A-Za-z0-9_-]+$")


class CanonicalWriteError(RuntimeError):
    """Raised when canonical partitions cannot be written and verified."""


@dataclass(frozen=True)
class PartitionWrite:
    """One verified division partition."""

    division: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class CanonicalWriteResult:
    """Structured summary of a canonical partition write."""

    input_row_count: int
    output_row_count: int
    partitions: tuple[PartitionWrite, ...]


def _verify_round_trip(expected: pd.DataFrame, temporary_path: Path) -> None:
    actual = pd.read_parquet(temporary_path, engine="pyarrow")
    actual = actual.loc[:, list(CANONICAL_COLUMNS)]
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_like=False)
    except AssertionError as exc:
        raise CanonicalWriteError(
            f"Parquet round-trip verification failed for {temporary_path.name}: {exc}"
        ) from exc


def write_canonical_partitions(
    frame: pd.DataFrame,
    destination_directory: Path,
) -> CanonicalWriteResult:
    """Write one verified Parquet file per present division.

    All partitions are written and verified through temporary files before any
    final destination is replaced.
    """

    ordered = sort_canonical_frame(frame)
    audit_canonical(ordered, input_row_count=len(frame)).raise_for_failure()
    destination = Path(destination_directory)
    divisions = sorted(ordered["Division"].unique().tolist())
    for division in divisions:
        if not _SAFE_PARTITION.fullmatch(str(division)):
            raise CanonicalWriteError(f"Unsafe division partition name: {division!r}")

    destination.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, pd.DataFrame, Path, Path]] = []
    temporary_paths: list[Path] = []
    try:
        for division in divisions:
            partition = (
                ordered.loc[ordered["Division"].eq(division), list(CANONICAL_COLUMNS)]
                .copy()
                .reset_index(drop=True)
            )
            final_path = destination / f"{division}.parquet"
            with tempfile.NamedTemporaryFile(
                prefix=f".{division}.",
                suffix=".parquet.tmp",
                dir=destination,
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

        for _, _, temporary_path, final_path in pending:
            os.replace(temporary_path, final_path)
            temporary_paths.remove(temporary_path)

        partitions = tuple(
            PartitionWrite(division, final_path, len(partition))
            for division, partition, _, final_path in pending
        )
        return CanonicalWriteResult(
            input_row_count=len(frame),
            output_row_count=sum(partition.row_count for partition in partitions),
            partitions=partitions,
        )
    except Exception:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise
