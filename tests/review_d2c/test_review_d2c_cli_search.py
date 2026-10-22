"""Review d2c: `recall search`'s early model check of commit 27dcf48. Fails on 27dcf48 for the reason in the
docstring."""

from __future__ import annotations

import time

import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_retrieval import load

from recall import cli, db
from recall.embedding import HashEmbedder


def test_search_does_not_hold_a_connection_idle_while_the_model_loads(
    schema: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """cmd_search now opens the connection, reads the stored model, and only then loads the model ("seconds, or a
    download", as 27dcf48 says) with that connection open and idle, then searches on it. Before, the model was
    loaded first and the connection opened after. A server or pooler that ends idle sessions (idle_session_timeout,
    a pgbouncer/cloud proxy idle limit, a NAT timeout) closes it during a slow load or a first-run download, and the
    search fails with "recall: can't use the database ...: terminating connection due to idle-session timeout"
    though the database and the model are both fine. Here the server ends sessions idle for 300 ms and the model
    takes 1 s to load. Correct: don't keep the connection across the load (check the stored model, close, load,
    then connect for search(), which checks again), or load first when the check passes on a short-lived
    connection."""
    with connect_to(schema) as c:
        db.migrate(c)
        load(c)
    monkeypatch.setenv(
        "RECALL_DATABASE_URL",
        f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public%20-cidle_session_timeout%3D300",
    )

    def slow_model(kind: str) -> HashEmbedder:
        time.sleep(1)
        return HashEmbedder()

    monkeypatch.setattr(cli, "make_embedder", slow_model)
    code = cli.main(["search", "pydantic", "--embedder", "hash", "--mode", "vector", "-k", "1"])
    assert capsys.readouterr().err == ""
    assert code == 0
