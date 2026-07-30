"""Small in-memory DuckDB query layer over canonical Parquet partitions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

MATCHES_VIEW = "matches"


def discover_canonical_files(directory: Path) -> tuple[Path, ...]:
    """Discover canonical Parquet files deterministically."""

    canonical_directory = Path(directory)
    if not canonical_directory.is_dir():
        raise FileNotFoundError(f"Canonical directory does not exist: {canonical_directory}")
    files = tuple(sorted(canonical_directory.glob("*.parquet"), key=lambda path: path.name))
    if not files:
        raise FileNotFoundError(
            f"Canonical directory contains no Parquet files: {canonical_directory}"
        )
    return files


def _sql_path_list(files: Sequence[Path]) -> str:
    quoted = ["'" + path.resolve().as_posix().replace("'", "''") + "'" for path in files]
    return "[" + ", ".join(quoted) + "]"


@contextmanager
def canonical_connection(directory: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield an in-memory connection with a union-by-name ``matches`` view."""

    files = discover_canonical_files(directory)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            f"CREATE VIEW {MATCHES_VIEW} AS "
            f"SELECT * FROM read_parquet({_sql_path_list(files)}, union_by_name = true)"
        )
        yield connection
    finally:
        connection.close()


def query_canonical(
    directory: Path,
    sql: str,
    parameters: Sequence[object] = (),
) -> pd.DataFrame:
    """Run one query and reliably close the temporary DuckDB connection."""

    with canonical_connection(directory) as connection:
        return connection.execute(sql, list(parameters)).df()
