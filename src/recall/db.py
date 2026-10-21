"""Connections and migrations. Migrations are the numbered .sql files in recall/migrations, applied once, in order."""

from __future__ import annotations

import os
import re
from importlib import resources
from importlib.resources.abc import Traversable

import psycopg
from psycopg import conninfo

DEFAULT_DATABASE_URL = "postgresql://localhost:5544/recall"

# Any number fits in a bigint advisory lock key; this one is "recall" in ASCII, so it's recognisable in pg_locks.
_MIGRATION_LOCK = 0x726563616C6C
_MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def database_url() -> str:
    return os.environ.get("RECALL_DATABASE_URL", DEFAULT_DATABASE_URL)


class BadDatabaseURL(ValueError):
    """The database URL can't be read as a connection string. Its text is not repeated: it may hold a password."""


def describe_url(url: str) -> str:
    """The URL's user, host, port and database, for messages: never its password."""
    try:
        parts = conninfo.conninfo_to_dict(url)
    except psycopg.ProgrammingError:
        return "(an unreadable RECALL_DATABASE_URL)"
    host = str(parts.get("host", ""))
    if "@" in host:
        # An '@' left unescaped in a password moves the rest of the password into the host.
        return "(RECALL_DATABASE_URL's host holds '@': write '@' in a password as %40)"
    user = f"{parts['user']}@" if parts.get("user") else ""
    port = f":{parts['port']}" if parts.get("port") else ""
    return " ".join(f"{user}{host}{port}/{parts.get('dbname', '')}".split())


def describe_error(url: str, error: Exception) -> str:
    """Why the database can't be used, on one line, without anything that could be part of the password."""
    try:
        parts = conninfo.conninfo_to_dict(url)
    except psycopg.ProgrammingError:
        return "it can't be read as a connection string"
    if "@" in str(parts.get("host", "")):
        return "check how the password is written"  # psycopg's message would quote the host, part of the password
    reason = " ".join(str(error).split())
    password = str(parts.get("password") or "")
    return reason.replace(password, "***") if password else reason


def connect(url: str | None = None, application_name: str = "recall") -> psycopg.Connection:
    """An autocommit connection: every transaction is an explicit `with conn.transaction()` block."""
    try:
        return psycopg.connect(url or database_url(), autocommit=True, application_name=application_name)
    except psycopg.ProgrammingError:
        raise BadDatabaseURL("the database URL can't be read as a connection string") from None


def lock_read_committed(conn: psycopg.Connection, key: int) -> None:
    """Begin the open transaction at READ COMMITTED and take the advisory lock `key` for it.

    The isolation level is set explicitly because waiting for the lock only works under READ COMMITTED: each
    statement after the wait sees what the previous holder committed. Under REPEATABLE READ (a database's or
    role's default can make it so) the snapshot is taken before the wait, and the second migrator or ingest fails
    on the first one's work. Must be the transaction's first statement.
    """
    conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (key,))


def migrations(directory: Traversable | None = None) -> list[tuple[str, str]]:
    """(version, sql) for every migration file, in version order. A badly named file is an error, not skipped."""
    found: list[tuple[str, str]] = []
    for entry in (directory or resources.files("recall.migrations")).iterdir():
        if not entry.name.endswith(".sql"):
            continue
        m = _MIGRATION_NAME.match(entry.name)
        if not m:
            raise ValueError(f"migration file name must look like 001_name.sql: {entry.name}")
        found.append((entry.name.removesuffix(".sql"), entry.read_text(encoding="utf-8")))
    found.sort()
    versions = [int(v[:3]) for v, _ in found]
    if versions != list(range(1, len(found) + 1)):
        raise ValueError(f"migrations must be numbered 001, 002, ... without gaps: {[v for v, _ in found]}")
    return found


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply the migrations not yet applied, each in its own transaction; return the versions applied.

    Concurrent migrators queue on an advisory lock, and each re-reads what's applied once it holds the lock.
    """
    if not conn.autocommit:
        raise ValueError("migrate needs an autocommit connection (recall.db.connect makes one)")
    applied_now: list[str] = []
    with conn.transaction():
        # Under the lock: two sessions creating the same table at once can both fail the IF NOT EXISTS check.
        lock_read_committed(conn, _MIGRATION_LOCK)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
    for version, sql in migrations():
        with conn.transaction():
            lock_read_committed(conn, _MIGRATION_LOCK)
            if conn.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,)).fetchone():
                continue
            conn.execute(sql.encode())
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied_now.append(version)
    return applied_now
