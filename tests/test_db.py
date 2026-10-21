from __future__ import annotations

import threading
from pathlib import Path

import psycopg
import pytest
from conftest import TEST_DATABASE_URL, connect_to

from recall import db


def test_migrations_are_numbered_in_order():
    versions = [v for v, _ in db.migrations()]
    assert versions[0] == "001_documents_and_chunks"
    assert versions == sorted(versions)


def test_migrate_applies_each_migration_once(schema: str):
    with connect_to(schema) as c:
        first = db.migrate(c)
        assert first == [v for v, _ in db.migrations()]
        assert db.migrate(c) == []
        rows = c.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert [r[0] for r in rows] == first


def test_concurrent_migrators_apply_each_migration_once(schema: str):
    applied: list[list[str]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def run() -> None:
        try:
            with connect_to(schema) as c:
                barrier.wait(timeout=10)
                applied.append(db.migrate(c))
        except BaseException as e:  # reported below, with the thread's error
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert sorted(len(a) for a in applied) == [0, 0, 0, len(db.migrations())]


def test_migrate_refuses_a_connection_that_is_not_autocommit():
    with psycopg.connect(TEST_DATABASE_URL) as c, pytest.raises(ValueError, match="autocommit"):
        db.migrate(c)


def insert_document(conn: psycopg.Connection, content: str = "abc") -> int:
    row = conn.execute(
        "INSERT INTO documents (path, title, url, content, content_sha256) VALUES ('a.md', 'A', 'u', %s, %s)"
        " RETURNING id",
        (content, "0" * 64),
    ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.parametrize(
    ("start", "end", "content", "tokens"),
    [
        (2, 2, "", 1),  # empty span
        (-1, 1, "ab", 1),  # negative start
        (0, 2, "abc", 1),  # content longer than its span
        (0, 1, "a", 0),  # no tokens
    ],
)
def test_chunk_rows_must_be_consistent(conn: psycopg.Connection, start: int, end: int, content: str, tokens: int):
    doc = insert_document(conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO chunks (document_id, ordinal, heading_path, start_offset, end_offset, content, token_count)"
            " VALUES (%s, 0, '{}', %s, %s, %s, %s)",
            (doc, start, end, content, tokens),
        )


def test_deleting_a_document_deletes_its_chunks(conn: psycopg.Connection):
    doc = insert_document(conn)
    conn.execute(
        "INSERT INTO chunks (document_id, ordinal, heading_path, start_offset, end_offset, content, token_count)"
        " VALUES (%s, 0, '{A}', 0, 3, 'abc', 1)",
        (doc,),
    )
    conn.execute("DELETE FROM documents WHERE id = %s", (doc,))
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,)


def test_the_vector_extension_is_usable(conn: psycopg.Connection):
    assert conn.execute("SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector").fetchone() == (0.0,)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["001_a.sql", "1_b.sql"], "must look like 001_name.sql"),
        (["001_a.sql", "002-b.sql"], "must look like 001_name.sql"),
        (["001_a.sql", "003_c.sql"], "without gaps"),
        (["002_b.sql"], "without gaps"),
        (["001_a.sql", "001_b.sql"], "without gaps"),
    ],
)
def test_badly_named_or_numbered_migrations_are_errors(tmp_path: Path, names: list[str], message: str):
    for name in names:
        (tmp_path / name).write_text("SELECT 1;")
    (tmp_path / "__init__.py").write_text("")  # not a migration: ignored
    with pytest.raises(ValueError, match=message):
        db.migrations(tmp_path)


def test_migrations_are_read_in_number_order(tmp_path: Path):
    for name in ["002_b.sql", "001_a.sql", "010_j.sql", *(f"00{i}_x.sql" for i in range(3, 10))]:
        (tmp_path / name).write_text(name)
    assert [v for v, _ in db.migrations(tmp_path)][:3] == ["001_a", "002_b", "003_x"]
    assert db.migrations(tmp_path)[-1] == ("010_j", "010_j.sql")


def test_two_chunks_of_a_document_cant_share_an_ordinal(conn: psycopg.Connection):
    doc = insert_document(conn)
    insert = (
        "INSERT INTO chunks (document_id, ordinal, heading_path, start_offset, end_offset, content, token_count)"
        " VALUES (%s, 0, '{}', %s, %s, %s, 1)"
    )
    conn.execute(insert, (doc, 0, 1, "a"))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(insert, (doc, 1, 2, "b"))


@pytest.mark.parametrize("commit", ["abc", "A" * 40, "g" * 40])
def test_the_settings_row_holds_a_full_lowercase_commit_id(conn: psycopg.Connection, commit: str):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO corpus_settings (source_commit, chunker_version, max_tokens) VALUES (%s, 1, 1)", (commit,)
        )
