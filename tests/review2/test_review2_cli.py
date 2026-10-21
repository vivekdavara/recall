"""Review 2 findings for recall.cli's database error message. Each test fails on commit 0e87c41."""

from __future__ import annotations

import socket

import pytest

from recall import cli


def unused_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_the_database_error_does_not_print_the_password(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """The message names the database by its URL, password included: 'recall migrate' with RECALL_DATABASE_URL
    'postgresql://alice:s3cret-pw@localhost:PORT/recall' prints s3cret-pw to stderr (and so into CI logs). The URL
    should be shown without its password (as psycopg's own message, which names host and port only, does)."""
    port = unused_port()
    monkeypatch.setenv("RECALL_DATABASE_URL", f"postgresql://alice:s3cret-pw@127.0.0.1:{port}/recall?connect_timeout=2")
    assert cli.main(["migrate"]) == 1
    assert "s3cret-pw" not in capsys.readouterr().err


def test_a_database_that_isnt_there_is_one_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """tests/test_cli.py's test_a_database_that_isnt_there_is_one_line_not_a_traceback only checks the start. When
    'localhost' resolves to ::1 and 127.0.0.1 (macOS, and Ubuntu runners), psycopg's message is four lines
    ("Multiple connection attempts failed. All failures were: ..."), and the .rstrip() keeps them all."""
    port = unused_port()
    monkeypatch.setenv("RECALL_DATABASE_URL", f"postgresql://localhost:{port}/recall?connect_timeout=2")
    if len({info[4][0] for info in socket.getaddrinfo("localhost", port, type=socket.SOCK_STREAM)}) < 2:
        pytest.skip("localhost has one address here, so psycopg reports one attempt")
    assert cli.main(["migrate"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1, err
