"""Review of ac9d86e (BM25 ranking): search_length, BM25's document length (migration 003_bm25.sql). Each test fails
on ac9d86e for the reason in its docstring."""

from __future__ import annotations

import psycopg
from test_bm25 import doc, occurrences, phrases, positions
from test_retrieval import load


def test_search_length_counts_every_occurrence_of_a_repeated_lexeme(conn: psycopg.Connection):
    """Defect: search_length drops occurrences past a lexeme's 255th, though 003_bm25.sql:11-12 says none is lost.

    003_bm25.sql:11 defines search_length as "How many lexemes a vector holds, counting every position: a lexeme that
    occurs three times counts three times", and :12 says a tsvector's cap on one lexeme's positions loses nothing
    because "a chunk is at most a few hundred words". But to_tsvector keeps at most 255 positions of a lexeme (256
    after the title's vector is concatenated), and the chunker bounds its own tokens, not lexemes: a token is `\\w+`
    (chunking.py:24), so x_x_..._x is one chunker token and 300 lexemes. BM25's length and tf both come from these
    truncated positions (retrieval.py _BM25: tf, search_length).

    Input: one page titled "Page" whose text is "x_x_..._x" (300 x's), ingested at the default 256 tokens per chunk:
    one chunk.
    Observed: search_length 256 (1 for the title + 255 x's).
    Expected: 301 (1 + 300), the lexeme count the comment and the BM25 formula describe; or the comment says the
    count is capped at 255 positions per lexeme and a chunk can reach the cap.

    Resolved the second way: the positions are gone from the tsvector, so neither the length nor tf can count them;
    the migration's comment and DESIGN.md "Storage" state the cap, and this test pins it, with tf capped alike.
    """
    load(conn, [doc("a.md", "_".join(["x"] * 300))], max_tokens=256)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
    assert conn.execute("SELECT search_length FROM chunks").fetchone() == (256,)  # 1 for the title + 255 x's
    # BM25's tf for x is the 255 positions kept, so a chunk's tf never exceeds its length.
    [(_, lexemes)] = phrases(conn, "x")
    [vector] = positions(conn).values()
    assert occurrences(vector, lexemes) == 255
