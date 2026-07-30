"""Credential-free, atomic acquisition of supported public source CSVs."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests

from fbai.data.sources import build_source_url, source_csv_path

DEFAULT_TIMEOUT: tuple[float, float] = (5.0, 30.0)
_CHUNK_SIZE = 64 * 1024


class AcquisitionError(RuntimeError):
    """Raised when a source CSV cannot be downloaded safely."""


@dataclass(frozen=True)
class SourceRequest:
    """One supported season/division download request."""

    season_code: str
    division: str


@dataclass(frozen=True)
class DownloadResult:
    """Successful atomic download metadata."""

    season_code: str
    division: str
    url: str
    path: Path
    bytes_written: int


@dataclass(frozen=True)
class DownloadFailure:
    """Structured per-file acquisition failure."""

    request: SourceRequest
    error_type: str
    message: str


@dataclass(frozen=True)
class BatchDownloadResult:
    """Results for a batch that may contain independent failures."""

    downloaded: tuple[DownloadResult, ...]
    failures: tuple[DownloadFailure, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failures


def _validate_timeout(timeout: tuple[float, float]) -> None:
    if len(timeout) != 2 or timeout[0] <= 0 or timeout[1] <= 0:
        raise ValueError("timeout must contain positive connect and read values")


def download_source_csv(
    *,
    season_code: str,
    division: str,
    destination_root: Path,
    overwrite: bool = False,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
) -> DownloadResult:
    """Download one public CSV through a temporary file and atomic replacement."""

    _validate_timeout(timeout)
    url = build_source_url(season_code, division)
    destination = source_csv_path(Path(destination_root), season_code, division)
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing source CSV without overwrite=True: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    own_session = session is None
    requester = session or requests.Session()
    response: requests.Response | None = None
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            response = requester.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            bytes_written = 0
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if not chunk:
                    continue
                temporary.write(chunk)
                bytes_written += len(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        if bytes_written == 0:
            raise AcquisitionError(f"Downloaded source CSV is empty: {url}")
        os.replace(temporary_path, destination)
        temporary_path = None
        return DownloadResult(
            season_code=season_code,
            division=division,
            url=url,
            path=destination,
            bytes_written=bytes_written,
        )
    except (FileExistsError, ValueError):
        raise
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to download {season_code}/{division}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if response is not None:
            response.close()
        if own_session:
            requester.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_sources(
    requests_to_download: Iterable[SourceRequest],
    *,
    destination_root: Path,
    overwrite: bool = False,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    session: requests.Session | None = None,
) -> BatchDownloadResult:
    """Download a batch and retain a structured failure for each failed file."""

    downloaded: list[DownloadResult] = []
    failures: list[DownloadFailure] = []
    for request in requests_to_download:
        try:
            downloaded.append(
                download_source_csv(
                    season_code=request.season_code,
                    division=request.division,
                    destination_root=destination_root,
                    overwrite=overwrite,
                    timeout=timeout,
                    session=session,
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch contract records each failure
            failures.append(
                DownloadFailure(
                    request=request,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
    return BatchDownloadResult(tuple(downloaded), tuple(failures))
