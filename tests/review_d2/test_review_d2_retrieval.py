"""Review d2 findings for recall.retrieval. Each test fails on commit 2fba151 for the reason in its docstring."""

from __future__ import annotations

import math
from collections.abc import Sequence

import psycopg
import pytest
from conftest import connect_to
from test_ingest import COMMIT
from test_retrieval import many_documents

from recall.embedding import HashEmbedder
from recall.ingest import ingest
from recall.retrieval import ModelMismatch, full_text, hybrid, search, vector


class Revised(HashEmbedder):
    """Another model whose vectors are close to the hash embedder's but not equal (as a new revision of a model's
    would be): each vector nudged along the last dimension and scaled back to unit length, under another name."""

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


def count_embedded(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()
    assert row is not None
    return int(row[0])


def test_vector_returns_k_rows_after_a_model_change(conn: psycopg.Connection):
    """vector() raises hnsw.ef_search to k so that "the vector list" can't "come back short". But an HNSW scan
    returns at most ef_search index entries, and the index still holds the entries of dead tuples (rows an ingest
    updated or deleted) until VACUUM; the heap drops the dead ones after the scan has picked its ef_search. A model
    change rewrites every chunk's embedding (SET NULL, then the new vector), so right after it about half of the
    nearest index entries are dead (old-model vectors): vector(k=60) over 80 live chunks returns 31 rows. (pgvector
    keeps identical vectors in one HNSW element, so the stand-in model must give vectors that differ.) It lasts until
    autovacuum gets to chunks. Correct: k rows whenever the corpus has at least k chunks (e.g. SET LOCAL
    hnsw.iterative_scan = strict_order, or VACUUM chunks after an ingest that wrote; both give 60 here).
    """
    docs = many_documents(80)
    ingest(conn, docs, source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    ingest(conn, docs, source_commit=COMMIT, max_tokens=64, embedder=Revised())
    assert count_embedded(conn) == 80
    assert len(vector(conn, Revised().embed_query("alpha beta"), k=60)) == 60


def test_vector_returns_k_rows_after_documents_are_replaced(conn: psycopg.Connection):
    """The same shortfall without a model change: an ingest that replaces documents deletes their chunks, whose
    index entries stay in the HNSW graph until VACUUM, and they use up the ef_search candidates. Here a corpus of 80
    pages is replaced by 80 others; vector(k=60) should return 60 of the 80 live chunks."""
    ingest(conn, many_documents(80, seed=7), source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    other = [d.__class__(f"n{d.path}", d.title, d.url, d.content + "\nomega\n") for d in many_documents(80, seed=8)]
    ingest(conn, other, source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    assert count_embedded(conn) == 80
    assert len(vector(conn, HashEmbedder().embed_query("alpha beta"), k=60)) == 60


def test_hybrid_vector_list_is_not_short_after_a_model_change(conn: psycopg.Connection):
    """hybrid's vector list has the same shortfall: with candidates=60 over 80 chunks after a model change, the
    question 'zzzz' (no full-text match) should still fuse 60 vector candidates, so k=60 hits come back."""
    docs = many_documents(80)
    ingest(conn, docs, source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    ingest(conn, docs, source_commit=COMMIT, max_tokens=64, embedder=Revised())
    assert full_text(conn, "zzzz") == []
    hits = hybrid(conn, "zzzz", Revised().embed_query("alpha beta"), k=60, candidates=60)
    assert len(hits) == 60


@pytest.mark.parametrize("k", [1001, 5000])
def test_vector_accepts_any_k_it_does_not_refuse(conn: psycopg.Connection, k: int):
    """_check_sizes accepts any k >= 1 and _hits sets hnsw.ef_search to max(k, 40), but pgvector's ef_search is at
    most 1000: vector(k=1001) fails with psycopg.errors.InvalidParameterValue ("1001 is outside the valid range for
    parameter hnsw.ef_search (1 .. 1000)"), and so does hybrid with candidates > 1000 and `recall search -k 1001
    --mode vector` (an uncaught traceback). Correct: either cap ef_search at 1000 (and use an iterative scan, or say
    the list may be short) or refuse k/candidates above 1000 with a ValueError that the CLI reports."""
    ingest(conn, many_documents(5), source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    v = HashEmbedder().embed_query("alpha")
    try:
        hits = vector(conn, v, k=k)
    except ValueError:
        return  # a clean refusal is acceptable
    assert len(hits) == 5


def test_hybrid_accepts_any_candidates_it_does_not_refuse(conn: psycopg.Connection):
    """As above, for hybrid(candidates=1001): InvalidParameterValue from set_config('hnsw.ef_search', '1001')."""
    ingest(conn, many_documents(5), source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    v = HashEmbedder().embed_query("alpha")
    try:
        hits = hybrid(conn, "alpha", v, k=10, candidates=1001)
    except ValueError:
        return
    assert len(hits) == 5


def test_vector_with_k_over_1000_in_a_fresh_session_is_not_silently_cut_to_40(schema: str, conn: psycopg.Connection):
    """The other face of the ef_search limit. In a session that hasn't used the vector type yet (every `recall
    search` process), pgvector's library isn't loaded when _hits runs set_config('hnsw.ef_search', '1001', true), so
    PostgreSQL stores an unchecked placeholder; when the query loads the library the out-of-range value is dropped
    with only a WARNING and ef_search is 40. The HNSW scan then returns 40 rows, silently: `recall search -k 1001
    --mode vector` over 1500 chunks prints 40. Correct: 1001 chunks, or a clean refusal."""
    ingest(conn, many_documents(1500), source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())
    v = HashEmbedder().embed_query("alpha beta")
    with connect_to(schema) as fresh:
        try:
            hits = vector(fresh, v, k=1001)
        except ValueError:
            return
    assert len(hits) == 1001


def test_search_never_compares_a_question_vector_with_another_models_vectors(schema: str, conn: psycopg.Connection):
    """search() calls check_model (one statement, its own snapshot), then embeds the question, then runs the query in
    a new transaction. An ingest with another model that commits in between (embedding a question with the real
    model takes tens of milliseconds) is not seen by the check but is seen by the query: the hash embedder's question
    vector is ranked against the other model's vectors, the mix the check exists to refuse. Correct: the model
    check and the retrieval read one snapshot (e.g. the check inside the query, or both in one REPEATABLE READ
    transaction), so this raises ModelMismatch (or returns hits from vectors of the question's model)."""
    docs = many_documents(10)
    ingest(conn, docs, source_commit=COMMIT, max_tokens=64, embedder=HashEmbedder())

    class Raced(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            with connect_to(schema) as other:
                ingest(other, docs, source_commit=COMMIT, max_tokens=64, embedder=Revised())
            return super().embed_query(text)

    with pytest.raises(ModelMismatch):
        search(conn, "alpha", Raced(), "vector", 3)


@pytest.mark.parametrize(
    "question",
    [
        " ".join(["pydantic"] * 20000),  # one distinct word: InternalError, "invalid memory alloc request size"
        " ".join(f"w{i}" for i in range(20000)) + " pydantic",  # likewise
        " ".join(f"w{i}" for i in range(40000)) + " pydantic",  # StatementTooComplex, "stack depth limit exceeded"
    ],
    ids=["repeated-word", "distinct-words", "deeper"],
)
def test_a_long_question_does_not_break_full_text_search(conn: psycopg.Connection, question: str):
    """Every whitespace-separated word becomes its own operand of one OR, duplicates included, with no limit.
    ts_rank_cd allocates one QueryRepresentationOperand (about 32 KB) per tsquery item, so a question of 20,000
    words (~160 KB of text, which fits in argv) asks PostgreSQL for ~1.3 GB and fails with InternalError "invalid
    memory alloc request size"; at ~40,000 words parsing the left-deep OR exceeds max_stack_depth. Even the
    same word repeated fails. Correct: any question returns hits (e.g. OR each distinct word once, and cap the
    number of words), as test_no_question_breaks_full_text_search claims for short ones."""
    from test_retrieval import load

    load(conn)
    hits = full_text(conn, question)
    # (Adjusted with the fix: only the first 64 distinct words are searched, so after 20,000 others "pydantic" is
    # past the limit and nothing matches; the point is that the search runs.)
    assert [h.path for h in hits] == (["body.md"] if question.startswith("pydantic") else [])
