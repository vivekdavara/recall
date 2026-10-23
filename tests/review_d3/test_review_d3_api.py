"""Review of day 3 (cf90395): the API's 500s where a 422 or 503 is meant."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_retrieval import load

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from recall import db
from recall.api import Services, create_app, make_pool
from recall.embedding import HashEmbedder
from recall.rerank import OverlapReranker


@pytest.fixture
def corpus(schema: str) -> str:
    with connect_to(schema) as conn:
        db.migrate(conn)
        load(conn)
    return schema


def client_for(schema: str, embedder: HashEmbedder | None) -> TestClient:
    def services() -> Services:
        configure = lambda c: c.execute(f"SET search_path TO {schema}, public")  # noqa: E731
        return Services(embedder, OverlapReranker(), None, make_pool(TEST_DATABASE_URL, configure, 2.0))

    return TestClient(create_app(services), raise_server_exceptions=False)


@pytest.fixture
def client(corpus: str) -> Iterator[TestClient]:
    with client_for(corpus, HashEmbedder()) as c:
        yield c


@pytest.mark.parametrize("path", ["/ask", "/search"])
def test_a_question_holding_a_nul_is_a_422_not_a_500(client: TestClient, path: str):
    """A question with a NUL character ("\\u0000", valid JSON) passes _question(), which only refuses lone surrogates,
    and psycopg then raises DataError ("PostgreSQL text fields cannot contain NUL (0x00) bytes"), which nothing maps:
    the response is a 500. Like the lone surrogate, it can't be searched, so it should be refused as a 422."""
    r = client.post(path, json={"question": "default \x00 values", "mode": "full-text"})
    assert r.status_code == 422, (r.status_code, r.text)


def test_without_an_embedder_a_vector_ask_is_not_a_500(corpus: str):
    """Services documents embedder=None as "only full-text search works", but /ask's default mode is hybrid, and
    retrieval.search's ValueError ("hybrid retrieval needs an embedder") reaches the client as a 500. A mode the
    server can't serve should be refused (a 422 or 503 saying so); /search has the same gap."""
    with client_for(corpus, None) as c:
        r = c.post("/ask", json={"question": "default values"})
        assert r.status_code in (422, 503), (r.status_code, r.text)
        r = c.post("/search", json={"question": "default values", "mode": "vector"})
        assert r.status_code in (422, 503), (r.status_code, r.text)
        assert c.post("/ask", json={"question": "default values", "mode": "full-text"}).status_code == 200
