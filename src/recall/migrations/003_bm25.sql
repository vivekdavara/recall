-- Day 5: what BM25 needs that ts_rank_cd doesn't: each chunk's length. DESIGN.md "Retrieval" is the specification.

-- The search vector's expression, as a function, so the length below is computed from the same vector. A generated
-- column can't read another generated column, and a copy of the expression could drift from it.
CREATE FUNCTION chunk_search_vector(heading_path text[], content text) RETURNS tsvector
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    RETURN setweight(to_tsvector('english', coalesce(heading_path[1], '')), 'A')
        || setweight(to_tsvector('english', coalesce(heading_text(heading_path[2:]), '')), 'B')
        || to_tsvector('english', content);

-- How many lexemes a vector holds, counting every position: a lexeme that occurs three times counts three times.
-- (A tsvector keeps at most 256 positions of one lexeme; a chunk is at most a few hundred words, so none is lost.)
CREATE FUNCTION tsvector_positions(v tsvector) RETURNS integer
    LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE
    RETURN (SELECT coalesce(sum(cardinality(l.positions)), 0) FROM unnest(v) AS l);

-- PostgreSQL 17: the same expression as before (002), now through the function. Rewrites the table.
ALTER TABLE chunks ALTER COLUMN search_vector SET EXPRESSION AS (chunk_search_vector(heading_path, content));

-- BM25's document length: the chunk's title, headings and text, in lexemes, stop words left out.
ALTER TABLE chunks
    ADD COLUMN search_length integer NOT NULL
        GENERATED ALWAYS AS (tsvector_positions(chunk_search_vector(heading_path, content))) STORED;
