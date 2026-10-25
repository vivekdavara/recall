"""The HTTP API on a small corpus in the test database: answers, search, health, validation and every 503."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import psycopg
import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_retrieval import DOCS, load

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool, PoolTimeout

from recall import db
from recall.answer import ask, build_request
from recall.api import Services, create_app, make_pool
from recall.embedding import HashEmbedder
from recall.llm import HAND_WRITTEN, Completion, RecordedProvider, Request, write_recording
from recall.rerank import OverlapReranker

QUESTION = "Which default values do skip and limit have?"


def in_schema(schema: str) -> Callable[[psycopg.Connection], None]:
    return lambda c: c.execute(f"SET search_path TO {schema}, public")


@pytest.fixture
def corpus(schema: str) -> str:
    with connect_to(schema) as conn:
        db.migrate(conn)
        load(conn)
    return schema


def client_for(
    schema: str | None,
    embedder: HashEmbedder | None = None,
    provider: RecordedProvider | None = None,
    url: str = TEST_DATABASE_URL,
    timeout: float = 5.0,
) -> TestClient:
    """An app whose services use the hash embedder, the overlap reranker and a pool into `schema`."""

    def services() -> Services:
        configure = in_schema(schema) if schema else None
        return Services(embedder or HashEmbedder(), OverlapReranker(), provider, make_pool(url, configure, timeout))

    return TestClient(create_app(services))


@pytest.fixture
def client(corpus: str) -> Iterator[TestClient]:
    with client_for(corpus) as c:
        yield c


def test_ask_gives_the_answer_its_citations_and_passages(client: TestClient):
    r = client.post("/ask", json={"question": QUESTION, "passages": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "extractive" and body["provider"] is None and body["problems"] == []
    assert body["answer"].startswith("skip and limit have default values. [1]")
    assert [p["number"] for p in body["passages"]] == [1, 2]
    assert body["passages"][0]["path"] == "query.md"
    citation = body["claims"][0]["citations"][0]
    page = next(d for d in DOCS if d.path == citation["path"])
    assert page.content[citation["start"] : citation["end"]] == citation["quote"]
    assert set(body["timings_ms"]) == {"retrieve", "rerank", "answer", "total"}


def test_ask_replays_a_recording_of_its_prompt(corpus: str, tmp_path: Path):
    # The passages ask() will answer from, so the recording is of the exact prompt the API sends.
    with connect_to(corpus) as conn:
        result, _ = ask(conn, QUESTION, HashEmbedder(), OverlapReranker(), None, passages=2)
    request = build_request(QUESTION, result.passages)
    text = json.dumps(
        {
            "answerable": True,
            "claims": [{"text": "They have defaults.", "citations": [{"passage": 1, "quote": "have default values"}]}],
        }
    )
    write_recording(tmp_path, request, Completion(text, HAND_WRITTEN), "test")
    with client_for(corpus, provider=RecordedProvider(tmp_path)) as c:
        body = c.post("/ask", json={"question": QUESTION, "passages": 2}).json()
        other = c.post("/ask", json={"question": "How do I send email after the response?"}).json()
    assert (body["method"], body["model"], body["provider"]) == ("llm", HAND_WRITTEN, "recorded")
    assert body["answer"] == "They have defaults. [1]"
    assert other["method"] == "extractive" and other["problems"][0].startswith("no recording")


def test_ask_without_the_reranker(client: TestClient):
    body = client.post("/ask", json={"question": QUESTION, "rerank": False, "passages": 3}).json()
    assert len(body["passages"]) == 3 and body["timings_ms"]["rerank"] == 0.0


def test_search_returns_hits_and_reranks_on_request(client: TestClient):
    hits = client.post("/search", json={"question": QUESTION, "k": 3}).json()["hits"]
    assert [h["number"] for h in hits] == [1, 2, 3] and "rerank_score" not in hits[0]
    ranked = client.post("/search", json={"question": QUESTION, "k": 3, "rerank": True}).json()["hits"]
    assert sorted(h["chunk_id"] for h in ranked) == sorted(h["chunk_id"] for h in hits)
    assert [h["rerank_score"] for h in ranked] == sorted((h["rerank_score"] for h in ranked), reverse=True)
    full_text = client.post("/search", json={"question": QUESTION, "mode": "full-text"}).json()["hits"]
    assert all(h["vector_rank"] is None for h in full_text)


def test_healthz_reports_the_corpus_and_the_models(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "documents": len(DOCS),
        "chunks": r.json()["chunks"],
        "stored_vectors": "hash-v1-384",
        "embedder": "hash-v1-384",
        "reranker": "overlap-v1",
        "provider": None,
    }
    assert r.json()["chunks"] >= len(DOCS)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   \n"},
        {"question": "x" * 2001},
        {"question": "q", "extra": 1},
        {"question": "q", "passages": 0},
        {"question": "q", "passages": 21},
        {"question": "q", "passages": 6, "depth": 5},
        {"question": "q", "depth": 101},
        {"question": "q", "mode": "tf-idf"},
        {"question": "q", "candidates": 1001},
        {"question": 5},
    ],
)
def test_a_bad_ask_is_a_422(client: TestClient, body: dict[str, Any]):
    assert client.post("/ask", json=body).status_code == 422


def test_passages_may_exceed_depth_without_the_reranker(client: TestClient):
    assert (
        client.post("/ask", json={"question": QUESTION, "passages": 6, "depth": 5, "rerank": False}).status_code == 200
    )


def test_a_question_with_a_lone_surrogate_is_a_422_not_a_500(client: TestClient):
    raw = '{"question": "default \\ud800 values"}'
    r = client.post("/ask", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert client.post("/search", content=raw, headers={"content-type": "application/json"}).status_code == 422
    # Where and why, but not the input itself (FastAPI's own handler echoed it, and couldn't encode this one).
    assert r.json()["detail"][0]["loc"] == ["body", "question"] and "input" not in r.json()["detail"][0]


def test_vectors_from_another_model_are_a_503(corpus: str):
    class Other(HashEmbedder):
        @property
        def name(self) -> str:
            return "another-model"

    with client_for(corpus, embedder=Other()) as c:
        r = c.post("/ask", json={"question": QUESTION})
        assert r.status_code == 503 and "another-model" in r.json()["detail"]
        health = c.get("/healthz")
        assert health.status_code == 503 and health.json()["status"] == "unavailable"
        # Full-text search needs no vectors.
        assert c.post("/search", json={"question": QUESTION, "mode": "full-text"}).status_code == 200


def test_a_database_with_no_corpus_is_a_503(schema: str):
    with client_for(schema) as c:
        for path in ("/ask", "/search"):
            r = c.post(path, json={"question": QUESTION})
            assert r.status_code == 503 and "run recall ingest" in r.json()["detail"], r.text
        assert c.get("/healthz").status_code == 503
    with connect_to(schema) as conn:
        db.migrate(conn)  # migrated, nothing ingested
    with client_for(schema) as c:
        r = c.post("/ask", json={"question": QUESTION})
        assert r.status_code == 503 and "no embedded corpus" in r.json()["detail"]


def test_no_database_is_a_503_that_says_nothing_of_the_url(monkeypatch: pytest.MonkeyPatch):
    url = "postgresql://someone:secret-password@127.0.0.1:1/recall"
    with client_for(None, url=url, timeout=0.5) as c:
        for r in (c.post("/ask", json={"question": QUESTION}), c.get("/healthz")):
            assert r.status_code == 503
            assert "unavailable" in r.text and "secret" not in r.text and "127.0.0.1" not in r.text


class PoolProbe:
    """Takes a connection from the app's one-connection pool while a model runs. It gets one only if the request
    running the model isn't holding the pool's only connection."""

    def __init__(self) -> None:
        self.pool: ConnectionPool | None = None
        self.got: list[str] = []

    def take(self, during: str) -> None:
        assert self.pool is not None
        try:
            with self.pool.connection(timeout=0.5):
                self.got.append(during)
        except PoolTimeout:
            self.got.append(f"no connection free during {during}")


def probing_client(schema: str, probe: PoolProbe) -> TestClient:
    class ProbingEmbedder(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            probe.take("embedding")
            return super().embed_query(text)

    class ProbingReranker(OverlapReranker):
        def score(self, question: str, texts: Sequence[str]) -> list[float]:
            probe.take("reranking")
            return super().score(question, texts)

    class ProbingProvider:
        name = "probe"

        def complete(self, request: Request) -> Completion:
            probe.take("the model call")
            return Completion(json.dumps({"answerable": False, "claims": []}), HAND_WRITTEN)

    def services() -> Services:
        probe.pool = make_pool(TEST_DATABASE_URL, in_schema(schema), size=1)
        return Services(ProbingEmbedder(), ProbingReranker(), ProbingProvider(), probe.pool)

    return TestClient(create_app(services))


def test_ask_holds_no_connection_while_a_model_runs(corpus: str):
    # The day-3 review: /ask held its pooled connection through the LLM call, so four slow answers used up the pool of
    # four and /search and /healthz failed with 503 "database unavailable" meanwhile.
    probe = PoolProbe()
    with probing_client(corpus, probe) as c:
        r = c.post("/ask", json={"question": QUESTION})
    assert r.status_code == 200 and r.json()["method"] == "none", r.text
    assert probe.got == ["embedding", "reranking", "the model call"]


def test_search_holds_no_connection_while_a_model_runs(corpus: str):
    probe = PoolProbe()
    with probing_client(corpus, probe) as c:
        r = c.post("/search", json={"question": QUESTION, "rerank": True})
    assert r.status_code == 200, r.text
    assert probe.got == ["embedding", "reranking"]


def test_asks_retrieve_time_includes_embedding_the_question(corpus: str):
    # The API embeds before taking a connection, so its own clock, not retrieve()'s, must cover the embedding.
    class SlowEmbedder(HashEmbedder):
        def embed_query(self, text: str) -> list[float]:
            time.sleep(0.05)
            return super().embed_query(text)

    with client_for(corpus, embedder=SlowEmbedder()) as c:
        timings = c.post("/ask", json={"question": QUESTION}).json()["timings_ms"]
    assert timings["retrieve"] >= 50
