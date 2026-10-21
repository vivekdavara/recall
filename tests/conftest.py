from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from recall import db

TEST_DATABASE_URL = os.environ.get("RECALL_TEST_DATABASE_URL", "postgresql://localhost:5544/recall_test")


@pytest.fixture
def schema() -> Iterator[str]:
    """A fresh schema in the test database, dropped afterwards. The vector extension lives in public."""
    name = f"t_{uuid.uuid4().hex[:12]}"
    try:
        admin = db.connect(TEST_DATABASE_URL, application_name="recall-tests")
    except psycopg.OperationalError as e:
        pytest.fail(f"no test database at {TEST_DATABASE_URL} (start the local one with scripts/pg.sh start): {e}")
    with admin:
        admin.execute(f"CREATE SCHEMA {name}")
        try:
            yield name
        finally:
            admin.execute(f"DROP SCHEMA {name} CASCADE")


def connect_to(schema: str) -> psycopg.Connection:
    conn = db.connect(TEST_DATABASE_URL, application_name=f"recall-tests-{schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    return conn


@pytest.fixture
def conn(schema: str) -> Iterator[psycopg.Connection]:
    """A connection to a migrated test schema."""
    with connect_to(schema) as c:
        db.migrate(c)
        yield c
