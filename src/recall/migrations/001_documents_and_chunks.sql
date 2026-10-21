-- Day 1: documents and their chunks. DESIGN.md "Storage" is the specification.

-- Day 2 stores embeddings with pgvector; creating the extension now makes a database without it fail on day 1.
-- It goes in public, so that a test schema can be dropped without dropping the extension.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

-- What the stored chunks were made from and with: one row. Ingest re-chunks every document when any of it changes.
CREATE TABLE corpus_settings (
    id boolean PRIMARY KEY DEFAULT true CHECK (id),
    source_commit text NOT NULL CHECK (source_commit ~ '^[0-9a-f]{40}$'),
    chunker_version integer NOT NULL CHECK (chunker_version >= 1),
    max_tokens integer NOT NULL CHECK (max_tokens >= 1),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE documents (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    path text NOT NULL UNIQUE CHECK (path <> ''),
    title text NOT NULL,
    url text NOT NULL,
    content text NOT NULL,
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    heading_path text[] NOT NULL,
    anchor text,
    start_offset integer NOT NULL CHECK (start_offset >= 0),
    end_offset integer NOT NULL CHECK (end_offset > start_offset),
    content text NOT NULL CHECK (char_length(content) = end_offset - start_offset),
    token_count integer NOT NULL CHECK (token_count >= 1),
    UNIQUE (document_id, ordinal)
);
