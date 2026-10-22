"""The real model on the real corpus: ingest it, then search it in every mode. A smoke test, not an evaluation.

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


def test_print_every_modes_rank_for_each_question(conn: psycopg.Connection, embedder: Any):
    lines = ["| question | page | full-text | vector | hybrid |", "|---|---|---|---|---|"]
    for question, page in ASSERTED + PRINTED_ONLY:
        ranks = [rank_of(retrieval.search(conn, question, embedder, m, 10), page) for m in retrieval.MODES]
        lines.append(f"| {question} | {page} | " + " | ".join(str(r) if r else "not in top 10" for r in ranks) + " |")
    table = "\n".join(lines)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### Smoke questions on the real corpus (rank of the page, top 10)\n\n" + table + "\n")
