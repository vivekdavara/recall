"""Ingest's embedding step and migration 002's columns (DESIGN.md "Embeddings" and "Storage")."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import psycopg
import pytest
from conftest import connect_to
from test_ingest import COMMIT, DOCS, chunk_ids

from recall import db
from recall import ingest as ingest_module
from recall.corpus.load import Document
from recall.db import vector_literal
from recall.embedding import HashEmbedder, passage_text
from recall.ingest import ingest


def run(conn: psycopg.Connection, docs: Sequence[Document], embedder: object = None, max_tokens: int = 16):  # type: ignore[no-untyped-def]
    return ingest(conn, docs, source_commit=COMMIT, max_tokens=max_tokens, embedder=embedder or HashEmbedder())


def vectors(conn: psycopg.Connection) -> dict[int, list[float] | None]:
    rows = conn.execute("SELECT id, embedding::text FROM chunks ORDER BY id").fetchall()
    return {i: None if e is None else [float(x) for x in e.strip("[]").split(",")] for i, e in rows}


class Renamed(HashEmbedder):
    """The same vectors under another model name: what a new model looks like to ingest."""

    @property
    def name(self) -> str:
        return "another-model"


class Revised(HashEmbedder):
    """Another model with vectors near the hash embedder's but not equal, as a new revision's would be. (Renamed's
    identical vectors hid the day-2 review's finding A: pgvector keeps identical vectors as one HNSW element.)"""

    @property
    def name(self) -> str:
        return "another-model"

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [nudge(v) for v in super().embed_passages(texts)]

    def embed_query(self, text: str) -> list[float]:
        return nudge(super().embed_query(text))


def nudge(v: list[float]) -> list[float]:
    w = [*v[:-1], v[-1] + 0.05]
    norm = math.sqrt(sum(x * x for x in w))
    return [x / norm for x in w]


class Failing(HashEmbedder):
    """Fails on its `fail_on`-th call (1-based)."""

    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.calls = 0
        self.fail_on = fail_on

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("model crashed")
        return super().embed_passages(texts)


def test_every_chunk_gets_the_vector_of_its_heading_path_and_text(conn: psycopg.Connection):
    report = run(conn, DOCS)
    rows = conn.execute("SELECT id, heading_path, content FROM chunks ORDER BY id").fetchall()
    assert report.chunks_embedded == len(rows) > 0
    stored = vectors(conn)
    for chunk_id, path, content in rows:
        expected = HashEmbedder().embed_passages([passage_text(path, content)])[0]
        # Stored as float4: equal to about 7 significant digits.
        assert stored[chunk_id] == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_only_new_chunks_are_embedded(conn: psycopg.Connection):
    run(conn, DOCS)
    before = vectors(conn)
    changed = [DOCS[0], replace(DOCS[1], content=DOCS[1].content + "\nMore beta.\n"), DOCS[2]]
    report = run(conn, changed)
    (b_chunks,) = conn.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.path = 'b.md'"
    ).fetchall()
    assert (report.chunks_written, report.chunks_embedded) == (b_chunks[0], b_chunks[0])
    after = vectors(conn)
    kept = before.keys() & after.keys()
    assert kept and all(after[i] == before[i] for i in kept)
    assert run(conn, changed).chunks_embedded == 0


def test_a_new_model_embeds_every_chunk_again_and_keeps_the_chunks(conn: psycopg.Connection):
    run(conn, DOCS)
    ids = chunk_ids(conn)
    report = run(conn, DOCS, Renamed())
    assert (report.unchanged, report.chunks_written, report.chunks_embedded) == (3, 0, len(ids))
    assert chunk_ids(conn) == ids
    assert conn.execute("SELECT embedding_model FROM corpus_settings").fetchone() == ("another-model",)
    assert run(conn, DOCS, Renamed()).chunks_embedded == 0
    assert run(conn, DOCS).chunks_embedded == len(ids)


def test_a_model_failing_part_way_leaves_the_previous_corpus_and_vectors(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
):
    run(conn, DOCS)
    before = vectors(conn)
    monkeypatch.setattr(ingest_module, "EMBED_BATCH", 2)
    with pytest.raises(RuntimeError, match="model crashed"):
        run(conn, DOCS, Failing(fail_on=2), max_tokens=8)
    assert vectors(conn) == before
    assert conn.execute("SELECT max_tokens, embedding_model FROM corpus_settings").fetchone() == (16, "hash-v1-384")


def test_a_model_of_another_dimension_is_refused(conn: psycopg.Connection):
    with pytest.raises(ValueError, match=r"gives 8 dimensions; chunks\.embedding has 384"):
        run(conn, DOCS, HashEmbedder(8))
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (0,)


def test_a_model_returning_too_few_vectors_is_refused(conn: psycopg.Connection):
    class Short(HashEmbedder):
        def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
            return super().embed_passages(texts)[1:]

    with pytest.raises(ValueError, match="returned"):
        run(conn, DOCS, Short())
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,)


def test_postgres_refuses_a_nan_vector(conn: psycopg.Connection):
    class NaN(HashEmbedder):
        def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
            return [[float("nan"), *v[1:]] for v in super().embed_passages(texts)]

    with pytest.raises(psycopg.DataError, match="NaN"):
        run(conn, DOCS, NaN())


def test_vector_literal_keeps_every_digit() -> None:
    assert vector_literal([0.1, -2.5e-08, 1]) == "[0.1,-2.5e-08,1.0]"


def test_the_search_vector_weighs_the_title_and_headings_above_the_text(conn: psycopg.Connection):
    page = Document("q.md", "Query Parameters", "u/q", "# Query Parameters\n\n## Defaults\n\nSkip and limit.\n")
    run(conn, [page])
    assert conn.execute("SELECT search_vector::text FROM chunks").fetchall() == [
        ("'default':3B 'limit':6 'paramet':2A 'queri':1A 'skip':4",)
    ]


def test_a_chunk_before_any_heading_has_only_text_in_its_search_vector(conn: psycopg.Connection):
    run(conn, [Document("c.md", "C", "u/c", "Gamma, before any heading.\n")])
    assert conn.execute("SELECT search_vector::text FROM chunks").fetchall() == [("'gamma':1 'head':4",)]


def test_migrating_a_day_one_database_keeps_its_chunks_and_the_next_ingest_embeds_them(schema: str):
    with connect_to(schema) as conn:
        version, sql = db.migrations()[0]
        conn.execute("CREATE TABLE schema_migrations (version text PRIMARY KEY, applied_at timestamptz DEFAULT now())")
        conn.execute(sql.encode())
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        # Day 1's ingest, as it was: documents, chunks and settings without a model.
        conn.execute(
            "INSERT INTO corpus_settings (source_commit, chunker_version, max_tokens) VALUES (%s, %s, 16)",
            (COMMIT, ingest_module.CHUNKER_VERSION),
        )
        doc_id = conn.execute(
            "INSERT INTO documents (path, title, url, content, content_sha256)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (DOCS[2].path, DOCS[2].title, DOCS[2].url, DOCS[2].content, DOCS[2].sha256),
        ).fetchone()
        assert doc_id is not None
        text = DOCS[2].content.strip()
        conn.execute(
            "INSERT INTO chunks (document_id, ordinal, heading_path, start_offset, end_offset, content,"
            " token_count) VALUES (%s, 0, '{}', 0, %s, %s, 6)",
            (doc_id[0], len(text), text),
        )
        assert db.migrate(conn) == ["002_retrieval"]
        assert conn.execute("SELECT embedding IS NULL, search_vector::text FROM chunks").fetchall() == [
            (True, "'gamma':1 'head':4")
        ]
        ids = chunk_ids(conn)
        report = run(conn, [DOCS[2]])
        assert (report.unchanged, report.chunks_written, report.chunks_embedded) == (1, 0, 1)
        assert chunk_ids(conn) == ids
        assert conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetchone() == (0,)


def test_the_indexes_exist(conn: psycopg.Connection):
    rows = conn.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema() AND tablename = 'chunks'"
        " AND indexname LIKE 'chunks_%%'"
        " ORDER BY indexname"
    ).fetchall()
    assert [(name, "hnsw" in d, "gin" in d) for name, d in rows if "embedding" in name or "search" in name] == [
        ("chunks_embedding_hnsw", True, False),
        ("chunks_search_vector_gin", False, True),
    ]


def test_dead_index_entries_from_an_ingest_dont_shorten_a_vector_list(conn: psycopg.Connection):
    """An iterative HNSW scan stops after hnsw.max_scan_tuples visited entries, dead ones included (the second day-2
    review: 21,000 deleted chunks near a question left an empty list). Ingest vacuums after a change, so the 1,000
    chunks it removed here are gone from the index. The limit is scaled down from 20,000 with the corpus: without
    the VACUUM a short list comes back here (3 or 4 rows of 10 in the runs I made)."""
    from recall.retrieval import vector

    conn.execute("ALTER TABLE chunks SET (autovacuum_enabled = false)")  # as if autovacuum hadn't got there yet
    near = [Document(f"n{i:04}.md", "N", "u", f"alpha beta n{i}\n") for i in range(1000)]
    far = [Document(f"f{i:03}.md", "F", "u", f"omega zeta f{i}\n") for i in range(50)]
    run(conn, near + far)
    run(conn, far)
    conn.execute("SET hnsw.max_scan_tuples = 1")
    assert len(vector(conn, HashEmbedder().embed_query("alpha beta"), k=10)) == 10


def test_an_ingest_that_changed_nothing_doesnt_vacuum(conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch):
    run(conn, DOCS)
    statements: list[str] = []
    real_execute = psycopg.Connection.execute

    def spy(self: psycopg.Connection, query: object, *args: object, **kwargs: object) -> object:
        statements.append(str(query))
        return real_execute(self, query, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(psycopg.Connection, "execute", spy)
    run(conn, DOCS)
    assert not any("VACUUM" in q for q in statements)
    run(conn, DOCS[:2])
    assert any("VACUUM" in q for q in statements)
