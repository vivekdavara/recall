"""The real models on the real corpus: ingest it, search it in every mode, rerank, answer. A smoke test, not an
evaluation.

Runs only when RECALL_REQUIRE_MODEL is set (CI's model job; locally `RECALL_REQUIRE_MODEL=1 pytest -m model`): the
ingest embeds 1,503 chunks, about 40 s on a laptop CPU. Needs the fetched corpus and the `embed` extra.

The questions are hand-written, a handful, and not a measure of quality: day 4's labelled set is. Three are asserted
(each mode puts the page in its top 3; a regression guard); all six are printed with every mode's rank, misses
included.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from conftest import TEST_DATABASE_URL, connect_to

from recall import db, retrieval
from recall.corpus import fetch
from recall.corpus.load import load
from recall.ingest import ingest

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not os.environ.get("RECALL_REQUIRE_MODEL"), reason="set RECALL_REQUIRE_MODEL=1 to run"),
]

CORPUS = Path(os.environ.get("RECALL_DATA_DIR", "data")) / "corpus"

ASSERTED = [
    ("Allow a frontend on another origin to call my API", "tutorial/cors.md"),
    ("response_model_exclude_unset", "tutorial/response-model.md"),
    ("How do I upload a file?", "tutorial/request-files.md"),
]
PRINTED_ONLY = [
    ("How do I declare a query parameter with a default value?", "tutorial/query-params.md"),
    ("How can I run a function after returning the response?", "tutorial/background-tasks.md"),
    ("How do I return a custom status code for an error?", "tutorial/handling-errors.md"),
]


@pytest.fixture(scope="module")
def embedder() -> Any:
    from recall.embedding import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder()


@pytest.fixture(scope="module")
def reranker() -> Any:
    from recall.rerank import CrossEncoderReranker

    return CrossEncoderReranker()


@pytest.fixture(scope="module")
def conn(embedder: Any) -> Iterator[psycopg.Connection]:
    manifest = fetch.load_manifest()
    documents = load(CORPUS, manifest)
    name = f"t_{uuid.uuid4().hex[:12]}"
    with db.connect(TEST_DATABASE_URL) as admin:
        admin.execute(f"CREATE SCHEMA {name}")
        try:
            with connect_to(name) as c:
                db.migrate(c)
                report = ingest(c, documents, source_commit=manifest.commit, max_tokens=256, embedder=embedder)
                assert (report.added, report.chunks_written, report.chunks_embedded) == (125, 1503, 1503)
                yield c
        finally:
            admin.execute(f"DROP SCHEMA {name} CASCADE")


def rank_of(hits: list[retrieval.Hit], page: str) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if h.path == page), None)


@pytest.mark.parametrize(("question", "page"), ASSERTED)
def test_every_mode_finds_the_page_in_its_top_3(conn: psycopg.Connection, embedder: Any, question: str, page: str):
    for mode in retrieval.MODES:
        hits = retrieval.search(conn, question, embedder, mode, 10)
        # Full-text returns only chunks holding a word of the question (6 hold response_model_exclude_unset).
        assert len(hits) == 10 or (mode == "full-text" and hits)
        rank = rank_of(hits, page)
        assert rank is not None and rank <= 3, (mode, [h.path for h in hits])


def test_every_stored_vector_is_the_models_vector_for_its_passage(conn: psycopg.Connection, embedder: Any):
    from recall.embedding import passage_text

    rows = conn.execute("SELECT id, heading_path, content, embedding::text FROM chunks ORDER BY id LIMIT 5").fetchall()
    fresh = embedder.embed_passages([passage_text(p, c) for _, p, c, _ in rows])
    for (_, _, _, stored), v in zip(rows, fresh, strict=True):
        assert [float(x) for x in stored.strip("[]").split(",")] == pytest.approx(v, abs=1e-6)


def reranked(conn: psycopg.Connection, question: str, embedder: Any, reranker: Any) -> list[retrieval.Hit]:
    """Hybrid's top 20, reranked: what ask() answers from."""
    from recall.rerank import DEFAULT_DEPTH, rerank

    return [
        r.hit for r in rerank(question, retrieval.search(conn, question, embedder, "hybrid", DEFAULT_DEPTH), reranker)
    ]


def test_print_every_modes_rank_for_each_question(conn: psycopg.Connection, embedder: Any, reranker: Any):
    lines = [
        "| question | page | full-text | vector | hybrid | hybrid + rerank |",
        "|---|---|---|---|---|---|",
    ]
    for question, page in ASSERTED + PRINTED_ONLY:
        ranks = [rank_of(retrieval.search(conn, question, embedder, m, 10), page) for m in retrieval.MODES]
        ranks.append(rank_of(reranked(conn, question, embedder, reranker)[:10], page))
        lines.append(f"| {question} | {page} | " + " | ".join(str(r) if r else "not in top 10" for r in ranks) + " |")
    table = "\n".join(lines)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### Smoke questions on the real corpus (rank of the page, top 10)\n\n" + table + "\n")


def test_a_top_10_vector_query_uses_the_hnsw_index_on_the_real_table(conn: psycopg.Connection, embedder: Any):
    """With retrieval's own settings for 10 rows, the planner must choose the index. On these 1,503 rows it chooses
    by cost between the index and a sequential scan: a scan once ef_search is about 80 to 100 (why EF_SEARCH_MIN stays
    at pgvector's default 40: at 100 this test fails), and already for hybrid's 50 candidates (ef_search 50) in a
    freshly analyzed table. Both give a full list; the scan is exact. So only the 10-row plan is pinned."""
    from recall.db import vector_literal

    v = vector_literal(embedder.embed_query("how do I give a query parameter a default value?"))
    # With the statistics autovacuum would gather: a table just loaded has none, and then the planner takes the index
    # whatever ef_search is (this test passed with the floor at 100 until it analyzed).
    conn.execute("ANALYZE chunks")
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(10, retrieval.EF_SEARCH_MIN)),))
        conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        sql = f"EXPLAIN WITH {retrieval._VECTOR} SELECT * FROM vec"
        plan = "\n".join(r[0] for r in conn.execute(sql.encode(), {"vector": v, "n": 10}))
    assert "Index Scan using chunks_embedding_hnsw" in plan, plan


@pytest.mark.parametrize(("question", "page"), ASSERTED)
def test_the_reranked_list_has_the_page_in_its_top_3(
    conn: psycopg.Connection, embedder: Any, reranker: Any, question: str, page: str
):
    hits = reranked(conn, question, embedder, reranker)
    rank = rank_of(hits, page)
    assert rank is not None and rank <= 3, [h.path for h in hits]


@pytest.mark.parametrize(("question", "page"), ASSERTED + PRINTED_ONLY)
def test_every_answer_cites_spans_of_its_pages(
    conn: psycopg.Connection, embedder: Any, reranker: Any, question: str, page: str
):
    from recall.answer import ask

    result, _ = ask(conn, question, embedder, reranker, None)
    assert result.method == "extractive" and len(result.passages) == 5 and result.claims
    pages = dict(conn.execute("SELECT path, content FROM documents").fetchall())
    for claim in result.claims:
        for c in claim.citations:
            assert pages[c.path][c.start : c.end] == c.quote
            assert c.path == result.passages[c.passage - 1].path


def test_print_the_extractive_answers(conn: psycopg.Connection, embedder: Any, reranker: Any):
    from recall.answer import ask

    lines = []
    for question, page in ASSERTED + PRINTED_ONLY:
        result, _ = ask(conn, question, embedder, reranker, None)
        cited = sorted({c.path for claim in result.claims for c in claim.citations})
        lines.append(f"- **{question}** (page: {page}; cited: {', '.join(cited)})\n  > {result.text}")
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n### Extractive answers on the real corpus (smoke questions)\n\n" + text + "\n")
