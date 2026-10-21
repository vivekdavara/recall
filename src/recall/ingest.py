"""Store documents and their chunks. DESIGN.md "Storage": one transaction, idempotent, re-chunk on new settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from recall.chunking import CHUNKER_VERSION, chunk_document
from recall.corpus.load import Document

# "ingest" in ASCII: ingests queue on this lock, so two never interleave.
_INGEST_LOCK = 0x696E67657374


@dataclass(frozen=True)
class IngestReport:
    added: int
    updated: int
    unchanged: int
    removed: int
    chunks_written: int


def ingest(
    conn: psycopg.Connection, documents: Sequence[Document], *, source_commit: str, max_tokens: int
) -> IngestReport:
    """Make the stored corpus exactly `documents`, chunked with max_tokens, in one transaction.

    A document whose content, title and URL are already stored, with chunks made by the same chunker and
    settings, is left alone; any other is inserted or updated and its chunks replaced. Stored documents not in
    `documents` are deleted with their chunks. Readers see the old corpus or the new one, never a mix.
    """
    paths = [d.path for d in documents]
    if len(set(paths)) != len(paths):
        raise ValueError("two documents with the same path")
    added = updated = unchanged = written = 0
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_INGEST_LOCK,))
        settings = (source_commit, CHUNKER_VERSION, max_tokens)
        stored = conn.execute("SELECT source_commit, chunker_version, max_tokens FROM corpus_settings").fetchone()
        same_settings = stored == settings
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
        conn.execute(
            "INSERT INTO corpus_settings (source_commit, chunker_version, max_tokens) VALUES (%s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE SET source_commit = excluded.source_commit,"
            " chunker_version = excluded.chunker_version, max_tokens = excluded.max_tokens, updated_at = now()",
            settings,
        )
    return IngestReport(added, updated, unchanged, removed, written)
