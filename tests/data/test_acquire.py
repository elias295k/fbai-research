from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

from fbai.data import acquire
from fbai.data.acquire import (
    AcquisitionError,
    SourceRequest,
    download_source_csv,
    download_sources,
)


class FakeResponse:
    def __init__(
        self,
        *,
        chunks: tuple[bytes, ...] = (b"Div,Date\n", b"E0,01/08/2023\n"),
        status_error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.status_error = status_error
        self.stream_error = stream_error
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, bool, tuple[float, float]]] = []
        self.closed = False

    def get(
        self,
        url: str,
        *,
        stream: bool,
        timeout: tuple[float, float],
    ) -> FakeResponse:
        self.calls.append((url, stream, timeout))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def forbid_real_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_session(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("A real requests.Session must not be created in tests")

    monkeypatch.setattr(acquire.requests, "Session", fail_session)


def test_successful_mocked_download_writes_expected_file(tmp_path: Path) -> None:
    response = FakeResponse()
    session = FakeSession([response])

    result = download_source_csv(
        season_code="2324",
        division="E0",
        destination_root=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    assert result.path == tmp_path / "2324" / "E0.csv"
    assert result.path.read_bytes() == b"Div,Date\nE0,01/08/2023\n"
    assert result.bytes_written == len(result.path.read_bytes())
    assert session.calls == [(result.url, True, (5.0, 30.0))]
    assert response.closed


def test_failed_response_does_not_replace_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "2324" / "E0.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"valid existing data")
    response = FakeResponse(status_error=requests.HTTPError("503"))

    with pytest.raises(AcquisitionError, match="HTTPError"):
        download_source_csv(
            season_code="2324",
            division="E0",
            destination_root=tmp_path,
            overwrite=True,
            session=FakeSession([response]),  # type: ignore[arg-type]
        )

    assert destination.read_bytes() == b"valid existing data"
    assert not list(destination.parent.glob("*.part"))


def test_partial_temporary_file_is_cleaned(tmp_path: Path) -> None:
    response = FakeResponse(
        chunks=(b"partial",),
        stream_error=requests.ConnectionError("connection interrupted"),
    )

    with pytest.raises(AcquisitionError, match="ConnectionError"):
        download_source_csv(
            season_code="2324",
            division="E0",
            destination_root=tmp_path,
            session=FakeSession([response]),  # type: ignore[arg-type]
        )

    destination = tmp_path / "2324" / "E0.csv"
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_overwrite_protection_avoids_network_call(tmp_path: Path) -> None:
    destination = tmp_path / "2324" / "E0.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")
    session = FakeSession([])

    with pytest.raises(FileExistsError, match="overwrite=True"):
        download_source_csv(
            season_code="2324",
            division="E0",
            destination_root=tmp_path,
            session=session,  # type: ignore[arg-type]
        )

    assert session.calls == []
    assert destination.read_bytes() == b"existing"


def test_batch_collects_per_file_failures(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(),
            FakeResponse(status_error=requests.HTTPError("404")),
        ]
    )

    result = download_sources(
        [SourceRequest("2324", "E0"), SourceRequest("2324", "D1")],
        destination_root=tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    assert len(result.downloaded) == 1
    assert len(result.failures) == 1
    assert not result.succeeded
    assert result.failures[0].request == SourceRequest("2324", "D1")
