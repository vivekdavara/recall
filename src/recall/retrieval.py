"""Retrieve chunks for a question: full-text search, vector search, and both fused by reciprocal rank fusion.

DESIGN.md "Retrieval" is the specification. Each retriever is one SQL query; the hybrid one runs both retrievers
as CTEs of a single query and fuses their ranks in SQL, so it reads one snapshot and makes one round trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import psycopg

from recall.db import vector_literal
from recall.embedding import Embedder

Mode = Literal["full-text", "vector", "hybrid"]
MODES: tuple[Mode, ...] = ("full-text", "vector", "hybrid")

DEFAULT_K = 10
# How many chunks each retriever hands to the fusion. A chunk outside both lists can't be fused in, so this bounds
# how deep in one list a chunk the other list ranks first can be found.
DEFAULT_CANDIDATES = 50
# The k in 1 / (k + rank), as in Cormack, Clarke and Buettcher (2009): large enough that rank 1 doesn't dominate.
RRF_K = 60
# pgvector's default hnsw.ef_search; an HNSW scan returns at most ef_search rows, so it is raised to the list length.
_EF_SEARCH_DEFAULT = 40


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    path: str
    title: str
    url: str  # the page's URL with the section's anchor
    heading_path: tuple[str, ...]
    start: int
    end: int
    content: str
    score: float  # full-text: ts_rank_cd; vector: cosine similarity; hybrid: the RRF sum
    full_text_rank: int | None  # 1-based rank in the full-text list (hybrid: None if not in it)
    vector_rank: int | None


class ModelMismatch(RuntimeError):
    """The stored vectors weren't made by the embedder asked to search them."""


# An OR of the question's words: plainto_tsquery ANDs them, so a chunk would have to hold every word. The text form
# of a tsquery quotes each lexeme and a lexeme never holds whitespace, so ' & ' can only be the operator.
_FULL_TEXT = """
fts AS (
    SELECT id, score, row_number() OVER (ORDER BY score DESC, id) AS rank
    FROM (
        SELECT c.id, ts_rank_cd(c.search_vector, q.query, 1) AS score
        FROM chunks c,
            (SELECT replace(plainto_tsquery('english', %(question)s)::text, ' & ', ' | ')::tsquery AS query) q
        WHERE c.search_vector @@ q.query
        ORDER BY score DESC, c.id
        LIMIT %(n)s
    ) ranked
)"""

# The inner ORDER BY distance LIMIT is the shape the HNSW index serves; numbering the rows happens outside it.
_VECTOR = """
vec AS (
    SELECT id, 1 - distance AS score, row_number() OVER (ORDER BY distance, id) AS rank
    FROM (
        SELECT c.id, c.embedding <=> %(vector)s::vector AS distance
        FROM chunks c
        ORDER BY c.embedding <=> %(vector)s::vector
        LIMIT %(n)s
    ) nearest
)"""

_FUSED = """
fused AS (
    SELECT coalesce(f.id, v.id) AS id,
        coalesce(1.0::float8 / (%(rrf_k)s + f.rank), 0) + coalesce(1.0::float8 / (%(rrf_k)s + v.rank), 0) AS score,
        f.rank AS fts_rank, v.rank AS vec_rank
    FROM fts f FULL JOIN vec v ON v.id = f.id
)"""

_HIT_COLUMNS = "c.id, d.path, d.title, d.url, c.anchor, c.heading_path, c.start_offset, c.end_offset, c.content"


def stored_model(conn: psycopg.Connection) -> str | None:
    row = conn.execute("SELECT embedding_model FROM corpus_settings").fetchone()
    return None if row is None else row[0]


def check_model(conn: psycopg.Connection, embedder: Embedder) -> None:
    stored = stored_model(conn)
    if stored is None:
        raise ModelMismatch("no embedded corpus: run recall ingest")
    if stored != embedder.name:
        raise ModelMismatch(
            f"the stored vectors are from {stored}, not {embedder.name}: search with that embedder, or ingest again"
        )


def full_text(conn: psycopg.Connection, question: str, k: int = DEFAULT_K) -> list[Hit]:
    """The k chunks whose search vectors best match any of the question's words, by ts_rank_cd."""
    _check_sizes(k, k)
    sql = f"WITH {_FULL_TEXT} SELECT {_HIT_COLUMNS}, f.score, f.rank, NULL FROM fts f {_JOIN} ORDER BY f.rank"
    return _hits(conn, sql, {"question": question, "n": k}, ef_search=None)


def vector(conn: psycopg.Connection, query_vector: Sequence[float], k: int = DEFAULT_K) -> list[Hit]:
    """The k chunks nearest the query vector by cosine distance, through the HNSW index (approximate)."""
    _check_sizes(k, k)
    sql = f"WITH {_VECTOR} SELECT {_HIT_COLUMNS}, v.score, NULL, v.rank FROM vec v {_JOIN.replace('f.id', 'v.id')}"
    return _hits(conn, sql + " ORDER BY v.rank", {"vector": vector_literal(query_vector), "n": k}, ef_search=k)


def hybrid(
    conn: psycopg.Connection,
    question: str,
    query_vector: Sequence[float],
    k: int = DEFAULT_K,
    *,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    """Both retrievers' top `candidates`, fused: score = sum over the lists a chunk is in of 1 / (rrf_k + rank).

    Ties (a chunk first in one list and another first in the other) go to the lower chunk id, so results repeat.
    """
    _check_sizes(k, candidates)
    if rrf_k < 0:
        raise ValueError(f"rrf_k must not be negative, not {rrf_k}")
    sql = (
        f"WITH {_FULL_TEXT}, {_VECTOR}, {_FUSED}"
        f" SELECT {_HIT_COLUMNS}, f.score, f.fts_rank, f.vec_rank FROM fused f {_JOIN}"
        " ORDER BY f.score DESC, f.id LIMIT %(k)s"
    )
    params = {"question": question, "vector": vector_literal(query_vector), "n": candidates, "k": k, "rrf_k": rrf_k}
    return _hits(conn, sql, params, ef_search=candidates)


def search(
    conn: psycopg.Connection,
    question: str,
    embedder: Embedder | None,
    mode: Mode = "hybrid",
    k: int = DEFAULT_K,
    *,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[Hit]:
    """Retrieve with one mode, embedding the question when the mode needs it."""
    if mode == "full-text":
        return full_text(conn, question, k)
    if embedder is None:
        raise ValueError(f"{mode} retrieval needs an embedder")
    check_model(conn, embedder)
    query_vector = embedder.embed_query(question)
    if mode == "vector":
        return vector(conn, query_vector, k)
    return hybrid(conn, question, query_vector, k, candidates=candidates)


_JOIN = "JOIN chunks c ON c.id = f.id JOIN documents d ON d.id = c.document_id"


def _check_sizes(k: int, candidates: int) -> None:
    if k < 1 or candidates < 1:
        raise ValueError(f"k and candidates must be at least 1, not {k} and {candidates}")


def _hits(conn: psycopg.Connection, sql: str, params: dict[str, Any], *, ef_search: int | None) -> list[Hit]:
    with conn.transaction():
        if ef_search is not None:
            # Local to this transaction. An HNSW scan returns at most ef_search rows: below the list length, the
            # vector list would come back short.
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(ef_search, _EF_SEARCH_DEFAULT)),))
        rows = conn.execute(sql.encode(), params).fetchall()
    return [
        Hit(
            chunk_id=r[0],
            path=r[1],
            title=r[2],
            url=f"{r[3]}#{r[4]}" if r[4] else r[3],
            heading_path=tuple(r[5]),
            start=r[6],
            end=r[7],
            content=r[8],
            score=float(r[9]),
            full_text_rank=r[10],
            vector_rank=r[11],
        )
        for r in rows
    ]
