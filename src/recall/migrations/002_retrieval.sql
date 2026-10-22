-- Day 2: vectors and full-text search over chunks. DESIGN.md "Retrieval" is the specification.

-- Which model made the stored vectors (its repository and revision, or the stub's name). NULL until an ingest has
-- embedded the chunks; when an ingest's model differs, it embeds every chunk again, so two models never mix.
ALTER TABLE corpus_settings ADD COLUMN embedding_model text CHECK (embedding_model <> '');

-- array_to_string is STABLE, because an element type's output can depend on settings (a timestamp's on DateStyle),
-- and a generated column may only call IMMUTABLE functions. The output of text is the text itself, which no setting
-- changes. A SQL-standard body is bound when the function is created, so the search path doesn't matter later.
CREATE FUNCTION heading_text(path text[]) RETURNS text
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
    RETURN array_to_string(path, ' ');

ALTER TABLE chunks
    -- Unit length (cosine distance is 1 - dot product); NULL only inside an ingest, before its embedding step.
    ADD COLUMN embedding vector(384),
    -- The page title (the first heading) weighs A, the headings under it B, the chunk text D: ts_rank_cd's default
    -- weights for those are 1.0, 0.4 and 0.1. 'english' stems words and drops stop words.
    ADD COLUMN search_vector tsvector NOT NULL GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(heading_path[1], '')), 'A')
        || setweight(to_tsvector('english', coalesce(heading_text(heading_path[2:]), '')), 'B')
        || to_tsvector('english', content)
    ) STORED;

-- HNSW with pgvector's defaults (m = 16, ef_construction = 64); queries set hnsw.ef_search.
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_search_vector_gin ON chunks USING gin (search_vector);
