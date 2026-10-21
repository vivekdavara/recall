"""Review findings for the advisory-lock queueing in db.migrate and ingest. Fails on 40a7144."""

from __future__ import annotations

import threading
from collections.abc import Callable

import psycopg
import pytest
from conftest import connect_to

from recall import db
from recall.corpus.load import Document
from recall.ingest import ingest

DOCS = [Document("a.md", "A", "u/a", "# A\n\nAlpha.\n"), Document("b.md", "B", "u/b", "Beta.\n")]


def concurrently(schema: str, work: Callable[[psycopg.Connection], object]) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        try:
            with connect_to(schema) as c:
                # A database or role configured with this default (ALTER DATABASE ... SET) is enough.
                c.execute("SET default_transaction_isolation = 'repeatable read'")
                barrier.wait(timeout=10)
                work(c)
        except BaseException as e:  # reported by the caller
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return errors


def test_concurrent_migrators_queue_even_when_the_default_isolation_is_repeatable_read(schema: str):
    """DESIGN.md: migrations are applied once, and concurrent migrators queue. The lock is taken by the first
    statement of the transaction, whose snapshot is taken before the wait, so under REPEATABLE READ the second
    migrator doesn't see the first's schema_migrations row and fails with DuplicateTable. The transactions
    should pin READ COMMITTED (or take the lock before the snapshot)."""
    assert concurrently(schema, db.migrate) == []


def test_concurrent_ingests_queue_even_when_the_default_isolation_is_repeatable_read(schema: str):
    """DESIGN.md: "Two ingests queue on an advisory lock ...; the second sees what the first committed." Under
    REPEATABLE READ the second doesn't, and fails with SerializationFailure on the documents upsert."""
    with connect_to(schema) as c:
        db.migrate(c)
    errors = concurrently(schema, lambda c: ingest(c, DOCS, source_commit="a" * 40, max_tokens=16))
    assert errors == []


pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
