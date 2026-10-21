from __future__ import annotations

import threading
from dataclasses import replace

import psycopg
import pytest
from conftest import connect_to

from recall import ingest as ingest_module
from recall.chunking import chunk_document
from recall.corpus.load import Document
from recall.ingest import IngestReport, ingest

COMMIT = "a" * 40

DOCS = [
    Document("a.md", "A", "u/a", "# A { #a }\n\nAlpha one.\n\n## Sub { #sub }\n\n" + "word " * 40 + "\n"),
    Document("b.md", "B", "u/b", "# B\n\n```python\n# not a heading\nx = 1\n```\n\nBeta.\n"),
    Document("c.md", "C", "u/c", "Gamma, before any heading.\n"),
]


def run(conn: psycopg.Connection, docs: list[Document], max_tokens: int = 16, commit: str = COMMIT) -> IngestReport:
    return ingest(conn, docs, source_commit=commit, max_tokens=max_tokens)


def stored(conn: psycopg.Connection) -> list[tuple[object, ...]]:
    return conn.execute(
        "SELECT d.path, c.ordinal, c.heading_path, c.anchor, c.start_offset, c.end_offset, c.content, c.token_count"
        " FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY d.path, c.ordinal"
    ).fetchall()


def expected(docs: list[Document], max_tokens: int = 16) -> list[tuple[object, ...]]:
    return [
        (d.path, c.ordinal, list(c.heading_path), c.anchor, c.start, c.end, c.text, c.token_count)
        for d in sorted(docs, key=lambda d: d.path)
        for c in chunk_document(d.content, max_tokens)
    ]


def chunk_ids(conn: psycopg.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id")]


def test_ingest_stores_every_chunk_as_a_span_of_its_document(conn: psycopg.Connection):
    report = run(conn, DOCS)
    assert report == IngestReport(added=3, updated=0, unchanged=0, removed=0, chunks_written=len(expected(DOCS)))
    assert stored(conn) == expected(DOCS)
    mismatched = conn.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE c.content <> substr(d.content, c.start_offset + 1, c.end_offset - c.start_offset)"
    ).fetchone()
    assert mismatched == (0,)
    assert conn.execute("SELECT source_commit, chunker_version, max_tokens FROM corpus_settings").fetchall() == [
        (COMMIT, ingest_module.CHUNKER_VERSION, 16)
    ]


def test_ingesting_the_same_corpus_again_writes_nothing(conn: psycopg.Connection):
    run(conn, DOCS)
    ids = chunk_ids(conn)
    assert run(conn, DOCS) == IngestReport(0, 0, 3, 0, 0)
    assert chunk_ids(conn) == ids


def test_a_changed_document_gets_new_chunks_and_the_others_keep_theirs(conn: psycopg.Connection):
    run(conn, DOCS)
    b_ids = conn.execute("SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE path = 'b.md'")
    kept = {r[0] for r in b_ids}
    changed = [DOCS[0], replace(DOCS[1], content=DOCS[1].content + "\nMore beta.\n"), DOCS[2]]
    report = run(conn, changed)
    assert (report.added, report.updated, report.unchanged, report.removed) == (0, 1, 2, 0)
    assert stored(conn) == expected(changed)
    now = {r[0] for r in conn.execute("SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id")}
    assert not kept & now
    assert conn.execute("SELECT content_sha256 FROM documents WHERE path = 'b.md'").fetchone() == (changed[1].sha256,)


@pytest.mark.parametrize("field", ["title", "url"])
def test_a_new_title_or_url_alone_updates_the_document(conn: psycopg.Connection, field: str):
    run(conn, DOCS)
    changed = [replace(DOCS[0], **{field: "new"}), *DOCS[1:]]
    assert run(conn, changed).updated == 1
    assert conn.execute(f"SELECT {field} FROM documents WHERE path = 'a.md'").fetchone() == ("new",)


def test_a_document_no_longer_in_the_corpus_is_removed_with_its_chunks(conn: psycopg.Connection):
    run(conn, DOCS)
    report = run(conn, DOCS[:2])
    assert (report.removed, report.unchanged) == (1, 2)
    assert stored(conn) == expected(DOCS[:2])


def test_an_empty_corpus_empties_the_store(conn: psycopg.Connection):
    run(conn, DOCS)
    assert run(conn, []).removed == 3
    assert stored(conn) == []


def test_new_settings_rechunk_every_document(conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch):
    run(conn, DOCS)
    assert run(conn, DOCS, max_tokens=8).updated == 3
    assert stored(conn) == expected(DOCS, max_tokens=8)
    assert run(conn, DOCS, max_tokens=8, commit="b" * 40).updated == 3
    monkeypatch.setattr(ingest_module, "CHUNKER_VERSION", ingest_module.CHUNKER_VERSION + 1)
    assert run(conn, DOCS, max_tokens=8, commit="b" * 40).updated == 3
    assert run(conn, DOCS, max_tokens=8, commit="b" * 40).unchanged == 3


def test_two_documents_with_one_path_are_refused_before_anything_is_written(conn: psycopg.Connection):
    with pytest.raises(ValueError, match="same path"):
        run(conn, [DOCS[0], DOCS[0]])
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (0,)


def test_a_failure_part_way_leaves_the_previous_corpus(conn: psycopg.Connection):
    run(conn, DOCS)
    before = stored(conn)
    # PostgreSQL text can't hold NUL: the third document fails after the first two were written.
    broken = [replace(DOCS[0], content="New alpha.\n"), DOCS[1], replace(DOCS[2], content="bad \x00 text\n")]
    with pytest.raises(psycopg.DataError):
        run(conn, broken, max_tokens=8)
    assert stored(conn) == before
    assert conn.execute("SELECT max_tokens FROM corpus_settings").fetchone() == (16,)


def test_concurrent_ingests_queue_and_the_second_finds_nothing_to_do(schema: str):
    with connect_to(schema) as c:
        from recall import db

        db.migrate(c)
    reports: list[IngestReport] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        try:
            with connect_to(schema) as c:
                barrier.wait(timeout=10)
                reports.append(run(c, DOCS))
        except BaseException as e:  # reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert sorted((r.added, r.unchanged) for r in reports) == [(0, 3), (3, 0)]
    with connect_to(schema) as c:
        assert stored(c) == expected(DOCS)
