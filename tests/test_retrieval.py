"""Retrieval (DESIGN.md "Retrieval") on small corpora embedded by the hash embedder, whose rankings are predictable."""

from __future__ import annotations

import random

import psycopg
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_ingest import COMMIT

from recall import retrieval
from recall.corpus.load import Document
from recall.embedding import HashEmbedder, passage_text
from recall.ingest import ingest
from recall.retrieval import Hit, ModelMismatch, full_text, hybrid, search, vector

DOCS = [
    Document(
        "query.md",
        "Query Parameters",
        "https://x/query/",
        "# Query Parameters { #query }\n\n## Defaults { #defaults }\n\nskip and limit have default values.\n\n"
        "## Optional { #optional }\n\nSet None to make one optional.\n",
    ),
    Document("body.md", "Request Body", "https://x/body/", "# Request Body\n\nSend JSON data with a Pydantic model.\n"),
    Document("cors.md", "CORS", "https://x/cors/", "# CORS\n\nCross-Origin Resource Sharing for another origin.\n"),
    Document(
        "tasks.md",
        "Background Tasks",
        "https://x/tasks/",
        "# Background Tasks\n\nRun a function after returning a response, such as sending email.\n",
    ),
    Document("bg.md", "Images", "https://x/bg/", "# Images\n\nA background image behind the docs.\n"),
]


def load(conn: psycopg.Connection, docs: list[Document] = DOCS, max_tokens: int = 64) -> None:
    ingest(conn, docs, source_commit=COMMIT, max_tokens=max_tokens, embedder=HashEmbedder())


def where(hits: list[Hit]) -> list[str]:
    return [h.url.removeprefix("https://x/") for h in hits]


def test_full_text_ranks_the_section_with_the_words_first(conn: psycopg.Connection):
    load(conn)
    hits = full_text(conn, "What are the default values?")
    assert where(hits)[0] == "query/#defaults"
    assert [h.full_text_rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.vector_rank is None for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_full_text_matches_any_word_not_every_word(conn: psycopg.Connection):
    load(conn)
    # No chunk holds both words: an AND query would find nothing.
    assert sorted(where(full_text(conn, "pydantic email"))) == ["body/", "tasks/"]


def test_a_word_in_the_title_outranks_the_same_word_in_the_text(conn: psycopg.Connection):
    load(conn)
    assert where(full_text(conn, "background")) == ["tasks/", "bg/"]


@pytest.mark.parametrize("question", ["", "what is it?", "   ", "!!!", "a & b | !c <-> (d)", "it's \\ 'quoted' :*"])
def test_questions_without_searchable_words_or_with_tsquery_syntax_are_safe(conn: psycopg.Connection, question: str):
    load(conn)
    hits = full_text(conn, question)
    assert len({h.chunk_id for h in hits}) == len(hits)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(question=st.text(max_size=60))
def test_no_question_breaks_full_text_search(conn: psycopg.Connection, question: str):
    if conn.execute("SELECT count(*) FROM documents").fetchone() == (0,):
        load(conn)
    if "\x00" in question:
        with pytest.raises(psycopg.DataError):
            full_text(conn, question)
        return
    full_text(conn, question)


def test_vector_search_finds_a_passage_by_its_own_text(conn: psycopg.Connection):
    load(conn)
    rows = conn.execute("SELECT id, heading_path, content FROM chunks ORDER BY id").fetchall()
    e = HashEmbedder()
    for chunk_id, path, content in rows:
        hits = vector(conn, e.embed_query(passage_text(path, content)), k=3)
        assert hits[0].chunk_id == chunk_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)
        assert [h.vector_rank for h in hits] == [1, 2, 3]
        assert all(h.full_text_rank is None for h in hits)


def many_documents(n: int, seed: int = 7) -> list[Document]:
    words = ["alpha", "beta", "gamma", "delta", "eta", "theta", "iota", "kappa", "mu", "nu", "xi", "pi", "sigma"]
    rng = random.Random(seed)
    return [
        Document(f"d{i:03}.md", f"Doc {i}", f"https://x/d{i}/", f"# Doc {i}\n\n{' '.join(rng.choices(words, k=12))}.\n")
        for i in range(n)
    ]


def test_vector_search_returns_k_rows_beyond_the_default_ef_search(conn: psycopg.Connection):
    load(conn, many_documents(80))
    # hnsw.ef_search defaults to 40, and an HNSW scan returns at most that many rows.
    assert len(vector(conn, HashEmbedder().embed_query("alpha beta"), k=60)) == 60


def fuse(ft: list[Hit], vec: list[Hit], k: int, rrf_k: int) -> list[tuple[int, float, int | None, int | None]]:
    """Reciprocal rank fusion in Python, from the two retrievers' own lists: the oracle for hybrid's SQL."""
    ranks: dict[int, list[int | None]] = {}
    for h in ft:
        ranks.setdefault(h.chunk_id, [None, None])[0] = h.full_text_rank
    for h in vec:
        ranks.setdefault(h.chunk_id, [None, None])[1] = h.vector_rank
    scores = {i: sum(1.0 / (rrf_k + r) if r is not None else 0.0 for r in rs) for i, rs in ranks.items()}
    order = sorted(scores, key=lambda i: (-scores[i], i))[:k]
    return [(i, scores[i], ranks[i][0], ranks[i][1]) for i in order]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    words=st.lists(
        st.sampled_from(["alpha", "beta", "gamma", "delta", "sigma", "doc", "7", "omega"]), min_size=1, max_size=4
    ),
    k=st.integers(1, 30),
    candidates=st.integers(1, 30),
    rrf_k=st.sampled_from([0, 1, 60]),
)
def test_hybrid_is_the_fusion_of_the_two_lists(
    conn: psycopg.Connection, words: list[str], k: int, candidates: int, rrf_k: int
):
    if conn.execute("SELECT count(*) FROM documents").fetchone() == (0,):
        load(conn, many_documents(40))
    question = " ".join(words)
    v = HashEmbedder().embed_query(question)
    expected = fuse(full_text(conn, question, candidates), vector(conn, v, candidates), k, rrf_k)
    got = hybrid(conn, question, v, k, candidates=candidates, rrf_k=rrf_k)
    assert [(h.chunk_id, h.full_text_rank, h.vector_rank) for h in got] == [(i, f, r) for i, _, f, r in expected]
    assert [h.score for h in got] == pytest.approx([s for _, s, _, _ in expected], rel=1e-12)


def test_hybrid_scores_a_chunk_in_one_list_by_that_rank_alone(conn: psycopg.Connection):
    load(conn)
    # "pydantic" is in body.md only; the hash vector of "zzz" is near nothing in particular.
    hits = hybrid(conn, "pydantic", HashEmbedder().embed_query("zzz"), k=10, candidates=5)
    body = next(h for h in hits if h.path == "body.md")
    assert body.full_text_rank == 1
    expected = 1 / 61 + (1 / (60 + body.vector_rank) if body.vector_rank else 0)
    assert body.score == pytest.approx(expected)


def test_hybrid_breaks_ties_by_chunk_id(conn: psycopg.Connection):
    load(conn)
    hits = hybrid(conn, "pydantic", HashEmbedder().embed_query("zzz"), k=10, candidates=1)
    assert len(hits) <= 2
    if len(hits) == 2 and hits[0].score == hits[1].score:
        assert hits[0].chunk_id < hits[1].chunk_id


def test_a_hit_points_at_its_span_and_section(conn: psycopg.Connection):
    load(conn)
    (hit,) = full_text(conn, "optional", k=1)
    assert (hit.path, hit.title, hit.url) == ("query.md", "Query Parameters", "https://x/query/#optional")
    assert hit.heading_path == ("Query Parameters", "Optional")
    content = DOCS[0].content
    assert content[hit.start : hit.end] == hit.content == "Set None to make one optional."


def test_a_chunk_before_any_heading_links_to_the_page(conn: psycopg.Connection):
    load(conn, [Document("p.md", "P", "https://x/p/", "Plain words only.\n")])
    assert where(full_text(conn, "plain")) == ["p/"]


def test_search_dispatches_by_mode(conn: psycopg.Connection):
    load(conn)
    e = HashEmbedder()
    q = "default values"
    assert search(conn, q, None, "full-text", 3) == full_text(conn, q, 3)
    assert search(conn, q, e, "vector", 3) == vector(conn, e.embed_query(q), 3)
    assert search(conn, q, e, "hybrid", 3, candidates=7) == hybrid(conn, q, e.embed_query(q), 3, candidates=7)


def test_vectors_are_only_searched_with_the_model_that_made_them(conn: psycopg.Connection):
    with pytest.raises(ModelMismatch, match="no embedded corpus"):
        search(conn, "x", HashEmbedder(), "vector")
    load(conn)

    class Other(HashEmbedder):
        @property
        def name(self) -> str:
            return "other"

    with pytest.raises(ModelMismatch, match="from hash-v1-384, not other"):
        search(conn, "x", Other(), "hybrid")
    with pytest.raises(ValueError, match="needs an embedder"):
        search(conn, "x", None, "vector")


@pytest.mark.parametrize(("k", "candidates", "rrf_k"), [(0, 5, 60), (5, 0, 60), (5, 5, -1)])
def test_sizes_are_checked(conn: psycopg.Connection, k: int, candidates: int, rrf_k: int):
    with pytest.raises(ValueError):
        hybrid(conn, "x", HashEmbedder().embed_query("x"), k, candidates=candidates, rrf_k=rrf_k)
    if rrf_k >= 0:
        with pytest.raises(ValueError):
            full_text(conn, "x", min(k, candidates))
        with pytest.raises(ValueError):
            vector(conn, HashEmbedder().embed_query("x"), min(k, candidates))


def plan(conn: psycopg.Connection, sql: str, params: dict[str, object]) -> str:
    with conn.transaction():
        # On a tiny table a sequential scan is cheaper; forbid it to see whether the index can serve the query.
        conn.execute("SET LOCAL enable_seqscan = off")
        rows = conn.execute(("EXPLAIN " + sql).encode(), params).fetchall()
    return "\n".join(r[0] for r in rows)


def test_the_vector_query_can_use_the_hnsw_index(conn: psycopg.Connection):
    load(conn)
    v = HashEmbedder().embed_query("x")
    sql = f"WITH {retrieval._VECTOR} SELECT * FROM vec"
    assert "Index Scan using chunks_embedding_hnsw" in plan(conn, sql, {"vector": retrieval.vector_literal(v), "n": 5})


def test_the_full_text_query_can_use_the_gin_index(conn: psycopg.Connection):
    load(conn)
    sql = f"WITH {retrieval._FULL_TEXT} SELECT * FROM fts"
    assert "chunks_search_vector_gin" in plan(conn, sql, {"question": "default values", "n": 5})


def test_an_identifier_matches_as_written_not_as_its_parts(conn: psycopg.Connection):
    load(
        conn,
        [
            Document("id.md", "Settings", "https://x/id/", "# Settings\n\nSet response_model_exclude_unset to skip.\n"),
            Document(
                "parts.md",
                "Response Model",
                "https://x/parts/",
                "# Response Model\n\nA response model can exclude fields, or unset ones; the response model again.\n",
            ),
        ],
    )
    # The parser splits the identifier into respons, model, exclud, unset; parts.md has all four, more often and in
    # its title, but not next to each other.
    assert where(full_text(conn, "response_model_exclude_unset")) == ["id/"]
    assert where(full_text(conn, "the response_model_exclude_unset parameter")) == ["id/"]
    # Separate words still match separately.
    assert where(full_text(conn, "response model exclude unset")) == ["parts/", "id/"]
