"""The HTTP API: POST /ask, POST /search and GET /healthz. The models load once, when the app starts.

DESIGN.md "The API" is the specification. Handlers are plain functions, which FastAPI runs on its thread pool; each
takes a connection from a small pool for the length of the request, and the models run one call at a time.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool, PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from recall import db, retrieval
from recall.answer import DEFAULT_PASSAGES, Answer, Timings, ask
from recall.embedding import Embedder
from recall.llm import Provider, provider_from_environment
from recall.rerank import DEFAULT_DEPTH, Reranker, rerank

# A question longer than this is refused (422): it is a question, not a document, and embedding a long one is slow.
MAX_QUESTION_CHARS = 2000
MAX_PASSAGES = 20
MAX_DEPTH = 100
POOL_SIZE = 4
POOL_TIMEOUT = 5.0  # seconds a request waits for a connection before a 503


class _OneAtATime:
    """Runs the models' calls one at a time. PyTorch already spreads one call over the CPU's cores, so concurrent
    calls would mostly take turns anyway; and Hugging Face's fast tokenizers have failed under concurrent use before
    ("Already borrowed", when two threads set truncation at once). 320 calls from 8 threads without the lock raised
    nothing here (tokenizers 0.23.3), so the lock is a precaution, not a fix for a failure seen."""

    def __init__(self, embedder: Embedder | None, reranker: Reranker) -> None:
        lock = threading.Lock()
        self.embedder = _LockedEmbedder(embedder, lock) if embedder is not None else None
        self.reranker = _LockedReranker(reranker, lock)


class _LockedEmbedder:
    def __init__(self, inner: Embedder, lock: threading.Lock) -> None:
        self._inner, self._lock = inner, lock

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_passages(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._inner.embed_query(text)


class _LockedReranker:
    def __init__(self, inner: Reranker, lock: threading.Lock) -> None:
        self._inner, self._lock = inner, lock

    @property
    def name(self) -> str:
        return self._inner.name

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        with self._lock:
            return self._inner.score(question, texts)


@dataclass
class Services:
    """What the handlers use: made once at startup, closed at shutdown."""

    embedder: Embedder | None  # None: only full-text search works
    reranker: Reranker
    provider: Provider | None
    pool: ConnectionPool

    def close(self) -> None:
        self.pool.close()


def services_from_environment() -> Services:
    """The models and provider named by the environment (RECALL_EMBEDDER, RECALL_RERANKER, RECALL_LLM), and a pool of
    connections to RECALL_DATABASE_URL."""
    from recall.cli import make_embedder
    from recall.rerank import make_reranker

    embedder = make_embedder(os.environ.get("RECALL_EMBEDDER", "model"))
    reranker = make_reranker(os.environ.get("RECALL_RERANKER", "model"))
    provider = provider_from_environment()
    return Services(embedder, reranker, provider, make_pool(db.database_url()))


def make_pool(
    url: str, configure: Callable[[psycopg.Connection], None] | None = None, timeout: float = POOL_TIMEOUT
) -> ConnectionPool:
    """Autocommit connections, as db.connect makes: every transaction is explicit. Opened without waiting, so the app
    starts while the database is down and answers 503 until it is up."""
    return ConnectionPool(
        url,
        kwargs={"autocommit": True, "application_name": "recall-api"},
        min_size=1,
        max_size=POOL_SIZE,
        timeout=timeout,
        configure=configure,
        open=True,
    )


def _question(value: str) -> str:
    if not value.strip():
        raise ValueError("the question is empty")
    if "\x00" in value:
        # Valid JSON ("\u0000"), but PostgreSQL text can't hold a NUL, so the query would fail (a 500).
        raise ValueError("the question holds a NUL character")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # JSON can carry a lone surrogate ("\ud800"), which PostgreSQL can't store or search.
        raise ValueError("the question isn't valid Unicode text") from None
    return value


Question = Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    mode: retrieval.Mode = "hybrid"
    passages: int = Field(DEFAULT_PASSAGES, ge=1, le=MAX_PASSAGES, description="how many passages to answer from")
    depth: int = Field(DEFAULT_DEPTH, ge=1, le=MAX_DEPTH, description="how many retrieved chunks to rerank")
    candidates: int = Field(retrieval.DEFAULT_CANDIDATES, ge=1, le=retrieval.MAX_K)
    rerank: bool = True

    @field_validator("question")
    @classmethod
    def _check_question(cls, value: str) -> str:
        return _question(value)

    @model_validator(mode="after")
    def _passages_within_depth(self) -> AskRequest:
        if self.rerank and self.passages > self.depth:
            raise ValueError(f"passages ({self.passages}) can't be more than depth ({self.depth})")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    mode: retrieval.Mode = "hybrid"
    k: int = Field(retrieval.DEFAULT_K, ge=1, le=retrieval.MAX_K)
    candidates: int = Field(retrieval.DEFAULT_CANDIDATES, ge=1, le=retrieval.MAX_K)
    rerank: bool = False

    @field_validator("question")
    @classmethod
    def _check_question(cls, value: str) -> str:
        return _question(value)


def _hit(number: int, hit: retrieval.Hit, rerank_score: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "number": number,
        "chunk_id": hit.chunk_id,
        "path": hit.path,
        "title": hit.title,
        "url": hit.url,
        "heading_path": list(hit.heading_path),
        "start": hit.start,
        "end": hit.end,
        "content": hit.content,
        "score": hit.score,
        "full_text_rank": hit.full_text_rank,
        "vector_rank": hit.vector_rank,
    }
    if rerank_score is not None:
        out["rerank_score"] = rerank_score
    return out


def _answer(result: Answer, timings: Timings) -> dict[str, Any]:
    return {
        "question": result.question,
        "answer": result.text,
        "method": result.method,
        "claims": [
            {
                "text": c.text,
                "citations": [
                    {
                        "passage": x.passage,
                        "chunk_id": x.chunk_id,
                        "path": x.path,
                        "url": x.url,
                        "quote": x.quote,
                        "start": x.start,
                        "end": x.end,
                    }
                    for x in c.citations
                ],
            }
            for c in result.claims
        ],
        "passages": [_hit(n, h) for n, h in enumerate(result.passages, 1)],
        "provider": result.provider,
        "model": result.model,
        "problems": list(result.problems),
        "timings_ms": {
            "retrieve": round(timings.retrieve, 2),
            "rerank": round(timings.rerank, 2),
            "answer": round(timings.answer, 2),
            "total": round(timings.total, 2),
        },
    }


class Unavailable(Exception):
    """The request can't be served now (503): no database, no corpus, or vectors from another model."""


def create_app(make_services: Callable[[], Services] = services_from_environment) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        services = make_services()
        state["services"] = services
        state["models"] = _OneAtATime(services.embedder, services.reranker)
        try:
            yield
        finally:
            services.close()

    app = FastAPI(title="Recall", version="0.3.0", lifespan=lifespan)

    @contextmanager
    def connection() -> Iterator[psycopg.Connection]:
        services: Services = state["services"]
        try:
            with services.pool.connection() as conn:
                yield conn
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            raise Unavailable("the database has no corpus or an older schema: run recall ingest") from None
        except retrieval.ModelMismatch as e:
            raise Unavailable(str(e)) from None
        except (psycopg.OperationalError, PoolTimeout):
            # Nothing of the URL or the server's message: either could hold part of a password.
            raise Unavailable("the database is unavailable") from None

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, e: RequestValidationError) -> JSONResponse:
        # FastAPI's own handler echoes each bad input back. An input holding a lone surrogate (JSON's "\ud800") can't
        # be encoded as UTF-8, so that echo failed and the 422 became a 500; and a 2,000-character question needn't
        # come back. Where and why is enough.
        errors = [{"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in e.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.exception_handler(Unavailable)
    async def unavailable(request: Request, e: Unavailable) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=503)

    def needs_embedder(mode: retrieval.Mode) -> None:
        if mode != "full-text" and state["models"].embedder is None:
            raise Unavailable(f"no embedding model is loaded, so only full-text search works, not {mode}")

    @app.post("/ask")
    def post_ask(body: AskRequest) -> dict[str, Any]:
        services: Services = state["services"]
        models: _OneAtATime = state["models"]
        needs_embedder(body.mode)
        with connection() as conn:
            result, timings = ask(
                conn,
                body.question,
                models.embedder,
                models.reranker,
                services.provider,
                mode=body.mode,
                passages=body.passages,
                depth=body.depth,
                candidates=body.candidates,
                use_reranker=body.rerank,
            )
        return _answer(result, timings)

    @app.post("/search")
    def post_search(body: SearchRequest) -> dict[str, Any]:
        models: _OneAtATime = state["models"]
        needs_embedder(body.mode)
        with connection() as conn:
            hits = retrieval.search(conn, body.question, models.embedder, body.mode, body.k, candidates=body.candidates)
        if not body.rerank:
            return {"hits": [_hit(n, h) for n, h in enumerate(hits, 1)]}
        ranked = rerank(body.question, hits, models.reranker)
        return {"hits": [_hit(n, r.hit, r.score) for n, r in enumerate(ranked, 1)]}

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        services: Services = state["services"]
        names = {
            "embedder": services.embedder.name if services.embedder else None,
            "reranker": services.reranker.name,
            "provider": services.provider.name if services.provider else None,
        }
        try:
            with connection() as conn, conn.transaction():
                row = conn.execute("SELECT (SELECT count(*) FROM documents), count(*) FROM chunks").fetchone()
                documents, chunks = row if row is not None else (0, 0)
                stored = retrieval.stored_model(conn)
        except Unavailable as e:
            return JSONResponse({"status": "unavailable", "detail": str(e), **names}, status_code=503)
        body = {"status": "ok", "documents": documents, "chunks": chunks, "stored_vectors": stored, **names}
        if services.embedder is not None and stored != services.embedder.name:
            body |= {"status": "unavailable", "detail": "the stored vectors aren't this embedder's: run recall ingest"}
            return JSONResponse(body, status_code=503)
        return JSONResponse(body)

    return app
