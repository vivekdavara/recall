"""Review 3 findings for db.describe_url and recall.cli's database error. Each test fails on commit 0777cfe."""

from __future__ import annotations

import socket

import pytest

from recall import cli, db


def unused_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_password_with_an_unescaped_at_sign_is_not_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A password with a raw '@' ('pa@ss-w0rd', a common mistake: it should be %40) makes libpq split the URL at
    the first '@', so the host is 'ss-w0rd@127.0.0.1'. describe_url prints 'alice@ss-w0rd@127.0.0.1:PORT/recall'
    and psycopg's reason repeats "failed to resolve host 'ss-w0rd@127.0.0.1'": most of the password reaches stderr.
    A host with '@' in it can only be a misread password: it should be withheld (e.g. the unreadable-URL text)."""
    port = unused_port()
    monkeypatch.setenv(
        "RECALL_DATABASE_URL", f"postgresql://alice:pa@ss-w0rd@127.0.0.1:{port}/recall?connect_timeout=2"
    )
    assert "w0rd" not in db.describe_url(db.database_url())
    assert cli.main(["migrate"]) == 1
    assert "w0rd" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "url",
    [
        # A key=value string whose password has an unquoted space: libpq's message quotes the password's tail.
        "host=127.0.0.1 port={port} dbname=recall user=alice password=open sesame",
        # A misspelt query parameter.
        "postgresql://alice:s3cret@127.0.0.1:{port}/recall?ssl_mode=require",
    ],
    ids=["unquoted-space", "bad-parameter"],
)
def test_an_unreadable_database_url_is_one_line_not_a_traceback(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """describe_url has a branch for an unreadable URL ("(an unreadable RECALL_DATABASE_URL)"), but cli.main only
    catches psycopg.OperationalError, and psycopg.connect raises psycopg.ProgrammingError for a URL it can't parse.
    So 'recall migrate' ends in a traceback whose last line is libpq's parse error, which for the first URL is
    'missing "=" after "sesame" in connection info string': part of the password. It should print one
    'recall: can't use the database ...' line without the password and return 1."""
    monkeypatch.setenv("RECALL_DATABASE_URL", url.format(port=unused_port()))
    assert cli.main(["migrate"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("recall: can't use the database ") and err.count("\n") == 1, err
    assert "sesame" not in err and "s3cret" not in err


def test_the_database_error_is_one_line_whatever_the_database_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """cli.py: "One line". The reason is joined onto one line, but describe_url's output isn't: a percent-encoded
    newline in the URL ('rec%0Aall') puts a line break in the message."""
    monkeypatch.setenv("RECALL_DATABASE_URL", f"postgresql://127.0.0.1:{unused_port()}/rec%0Aall?connect_timeout=2")
    assert cli.main(["migrate"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1, err
