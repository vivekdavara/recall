"""Store documents, their chunks and the chunks' vectors. DESIGN.md "Storage": one transaction, idempotent, re-chunk
on new settings, re-embed on a new model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from recall import db
from recall.chunking import CHUNKER_VERSION, chunk_document
from recall.corpus.load import Document
from recall.embedding import DIMENSION, Embedder, passage_text

# "ingest" in ASCII: ingests queue on this lock, so two never interleave.
_INGEST_LOCK = 0x696E67657374

# Chunks embedded per model call and per UPDATE.
EMBED_BATCH = 64


@dataclass(frozen=True)
class IngestReport:
    added: int
    updated: int
    unchanged: int
    removed: int
    chunks_written: int
    chunks_embedded: int


def ingest(
    conn: psycopg.Connection,
    documents: Sequence[Document],
    *,
    source_commit: str,
    max_tokens: int,
    embedder: Embedder,
) -> IngestReport:
    """Make the stored corpus exactly `documents`, chunked with max_tokens and embedded by `embedder`, in one
    transaction.

    A document whose content, title and URL are already stored, with chunks made by the same chunker and
    settings, is left alone; any other is inserted or updated and its chunks replaced. Stored documents not in
    `documents` are deleted with their chunks. Then every chunk without a vector is embedded: the new ones, or all
    of them when the stored vectors came from another model. Readers see the old corpus or the new one, never a mix.
    """
    paths = [d.path for d in documents]
    if len(set(paths)) != len(paths):
        raise ValueError("two documents with the same path")
    added = updated = unchanged = written = 0
    with conn.transaction():
        db.lock_read_committed(conn, _INGEST_LOCK)
        settings = (source_commit, CHUNKER_VERSION, max_tokens)
        stored = conn.execute(
            "SELECT source_commit, chunker_version, max_tokens, embedding_model FROM corpus_settings"
        ).fetchone()
        same_settings = stored is not None and stored[:3] == settings
        if stored is not None and stored[3] != embedder.name:
            conn.execute("UPDATE chunks SET embedding = NULL")
        existing = {
            row[0]: tuple(row[1:]) for row in conn.execute("SELECT path, content_sha256, title, url FROM documents")
        }
        for doc in documents:
            before = existing.get(doc.path)
            if same_settings and before == (doc.sha256, doc.title, doc.url):
                unchanged += 1
                continue
            row = conn.execute(
                "INSERT INTO documents (path, title, url, content, content_sha256) VALUES (%s, %s, %s, %s, %s)"
                " ON CONFLICT (path) DO UPDATE SET title = excluded.title, url = excluded.url,"
                " content = excluded.content, content_sha256 = excluded.content_sha256, ingested_at = now()"
                " RETURNING id",
                (doc.path, doc.title, doc.url, doc.content, doc.sha256),
            ).fetchone()
            assert row is not None
            document_id = row[0]
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            chunks = chunk_document(doc.content, max_tokens)
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, ordinal, heading_path, anchor, start_offset, end_offset,"
                    " content, token_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (document_id, c.ordinal, list(c.heading_path), c.anchor, c.start, c.end, c.text, c.token_count)
                        for c in chunks
                    ],
                )
            written += len(chunks)
            if before is None:
                added += 1
            else:
                updated += 1
        removed = conn.execute("DELETE FROM documents WHERE path <> ALL (%s)", (paths,)).rowcount
        embedded = _embed_missing(conn, embedder)
        conn.execute(
            "INSERT INTO corpus_settings (source_commit, chunker_version, max_tokens, embedding_model)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET source_commit = excluded.source_commit,"
            " chunker_version = excluded.chunker_version, max_tokens = excluded.max_tokens,"
            " embedding_model = excluded.embedding_model, updated_at = now()",
            (*settings, embedder.name),
        )
    return IngestReport(added, updated, unchanged, removed, written, embedded)


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text form, '[x,y,...]'. repr keeps every digit, so nothing is lost before PostgreSQL's float4."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _embed_missing(conn: psycopg.Connection, embedder: Embedder) -> int:
    """Embed every chunk that has no vector, in id order; return how many."""
    if embedder.dimension != DIMENSION:
        raise ValueError(f"{embedder.name} gives {embedder.dimension} dimensions; chunks.embedding has {DIMENSION}")
    rows = conn.execute("SELECT id, heading_path, content FROM chunks WHERE embedding IS NULL ORDER BY id").fetchall()
    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i : i + EMBED_BATCH]
        vectors = embedder.embed_passages([passage_text(path, content) for _, path, content in batch])
        if len(vectors) != len(batch):
            raise ValueError(f"{embedder.name} returned {len(vectors)} vectors for {len(batch)} passages")
        conn.execute(
            "UPDATE chunks SET embedding = v.embedding::vector FROM unnest(%s::bigint[], %s::text[]) AS"
            " v(id, embedding) WHERE chunks.id = v.id",
            ([row[0] for row in batch], [vector_literal(v) for v in vectors]),
        )
    return len(rows)
