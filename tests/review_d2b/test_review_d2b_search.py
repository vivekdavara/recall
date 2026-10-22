"""Review d2b: search()'s one-transaction fix of commit 9c93d03. Fails on 3a4673c for the reason in the docstring."""

from __future__ import annotations

import psycopg
import pytest
from test_retrieval import load

from recall.embedding import HashEmbedder
from recall.retrieval import search


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_search_inside_a_callers_transaction_is_refused_cleanly(conn: psycopg.Connection, mode: str):
    """search() says it "needs an autocommit connection" because the model check and the query must run "in one
    transaction of their own", and it guards that with `if not conn.autocommit`. But a psycopg autocommit connection
    inside the caller's `with conn.transaction():` still has autocommit True: search's conn.transaction() is then a
    savepoint, and its SET TRANSACTION ISOLATION LEVEL REPEATABLE READ fails with psycopg.errors.ActiveSqlTransaction
    ("must be called before any query") — or, if the caller's block has run nothing yet, "must not be called in a
    subtransaction" — instead of the ValueError the guard exists to give. Correct: refuse with the ValueError whenever
    search can't open its own transaction (check conn.info.transaction_status is IDLE, not only conn.autocommit), or
    run in the caller's transaction."""
    load(conn)
    with conn.transaction():
        conn.execute("SELECT 1")
        with pytest.raises(ValueError, match=r"autocommit|transaction"):
            search(conn, "pydantic", HashEmbedder(), mode, 3)
