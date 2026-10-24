"""Review d2c: ingest's VACUUM (ANALYZE) of commit b50239e. Fails on 27dcf48 for the reasons in the docstrings."""

from __future__ import annotations

import psycopg
import pytest
from conftest import connect_to
from psycopg import pq
from test_ingest import COMMIT, DOCS

from recall.embedding import HashEmbedder
from recall.ingest import ingest


def test_ingest_on_a_non_autocommit_connection_does_not_fail_after_committing(conn: psycopg.Connection):
    """ingest() vacuums when `conn.info.transaction_status == IDLE`, taking "idle" to mean "VACUUM can run". On a
    connection with autocommit off (psycopg's default, and the kind fe2f75f made search() accept), idle is also what
    the connection is after ingest's own `with conn.transaction()` commits, but psycopg then sends BEGIN before the
    next statement, so the VACUUM raises ActiveSqlTransaction "VACUUM cannot run inside a transaction block". The
    ingest has already committed: the caller gets an exception for an ingest that succeeded, no IngestReport, and a
    connection left in a failed transaction (INERROR). Before b50239e this ingest returned its report. Correct: run
    the VACUUM only when it can run (idle and autocommit, or switch autocommit on for it), and never let the
    after-commit maintenance turn a committed ingest into an error."""
    conn.autocommit = False
    report = ingest(conn, DOCS, source_commit=COMMIT, max_tokens=16, embedder=HashEmbedder())
    assert report.added == len(DOCS)
    assert conn.info.transaction_status == pq.TransactionStatus.IDLE


def test_a_vacuum_that_cannot_get_its_lock_does_not_fail_a_committed_ingest(schema: str, conn: psycopg.Connection):
    """VACUUM needs a SHARE UPDATE EXCLUSIVE lock on both tables, which the ingest's own writes (ROW EXCLUSIVE) don't:
    another session's ANALYZE, VACUUM, CREATE INDEX CONCURRENTLY or an explicit lock in that mode lets the ingest
    commit but makes its VACUUM wait, indefinitely, or under a lock_timeout (or statement_timeout) fail. The failure
    comes after the commit, so ingest() raises LockNotAvailable for a corpus it has replaced (`recall ingest` then
    prints "can't use the database ...: canceling statement due to lock timeout" and exits 1), and running it again
    changes nothing, so it doesn't vacuum again either. Correct: the VACUUM is best effort after the commit; a failure
    of it is reported as a warning (or ignored), and ingest returns its report."""
    with connect_to(schema) as blocker, blocker.transaction():
        blocker.execute("LOCK TABLE chunks IN SHARE UPDATE EXCLUSIVE MODE")
        conn.execute("SET lock_timeout = '200ms'")
        with pytest.warns(UserWarning, match="the ingest is committed, but VACUUM after it failed: .*lock timeout"):
            report = ingest(conn, DOCS, source_commit=COMMIT, max_tokens=16, embedder=HashEmbedder())
    assert report.added == len(DOCS)


@pytest.fixture(autouse=True)
def _reset(conn: psycopg.Connection):
    yield
    if conn.info.transaction_status == pq.TransactionStatus.INERROR:
        conn.rollback()
