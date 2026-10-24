"""Retrieve chunks for a question: full-text search, vector search, and both fused by reciprocal rank fusion.

DESIGN.md "Retrieval" is the specification. Each retriever is one SQL query; the hybrid one runs both retrievers
as CTEs of a single query and fuses their ranks in SQL, so it reads one snapshot and makes one round trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from psycopg import pq

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
# The smallest hnsw.ef_search a query uses: pgvector's default. Raising it to 100 looked like exact top-10 results,
# but on this corpus's 1,503 rows the planner switches to a sequential scan once ef_search is about 80 to 100, so that
# was an exact scan, not HNSW (scripts/measure_retrieval.py now forces and checks each plan).
EF_SEARCH_MIN = 40
# pgvector's largest ef_search. A list is at most this long, so ef_search can always be raised to the list length.
MAX_K = 1000
# What full-text search uses of a question (see _FULL_TEXT): its first 64 distinct words, a word's phrase only if it
# has at most 16 lexemes (31 query nodes), and words only while the query has at most 256 nodes in all.
MAX_QUESTION_WORDS = 64
MAX_PHRASE_NODES = 31
MAX_QUERY_NODES = 256
_TEXT_LIMITS = {"max_words": MAX_QUESTION_WORDS, "max_phrase_nodes": MAX_PHRASE_NODES, "max_nodes": MAX_QUERY_NODES}


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


# An OR over the question's words (split at whitespace), each word a phrase of its lexemes. OR, because
# plainto_tsquery ANDs every word, so a chunk would have to hold all of them. A phrase, because the parser splits an
# identifier: response_model_exclude_unset is four lexemes, and ORing those four ranks every chunk about response
# models above the one about that parameter; as a phrase they must be adjacent, as they are in the identifier. A
# word that is only stop words (or punctuation) has no lexemes and is left out. The text form of a tsquery quotes
# each lexeme, so wrapping each phrase in parentheses and joining with ' | ' can't change what a lexeme says. The
# query's size is bounded (the limits above): ts_rank_cd allocates about 32 KB per query node, so a question of
# 20,000 words asked PostgreSQL for 1.3 GB and failed, and one word with no spaces ("0,1,2,...,19999", a pasted
# list) is a phrase of 39,999 nodes, whose text overflowed the stack when cast back to a tsquery (the day-2
# reviews). So each distinct phrase counts once, a phrase over MAX_PHRASE_NODES is skipped, and words are taken in
# order while the query's running count of nodes, its ORs included, stays within MAX_QUERY_NODES, at most
# MAX_QUESTION_WORDS of them.
_FULL_TEXT = """
fts AS (
    SELECT id, score, row_number() OVER (ORDER BY score DESC, id) AS rank
    FROM (
        SELECT c.id, ts_rank_cd(c.search_vector, q.query, 1) AS score
        FROM chunks c, (
            SELECT string_agg('(' || phrase::text || ')', ' | ' ORDER BY n)::tsquery AS query
            FROM (
                -- Each phrase's nodes and the OR before it (the first has none).
                SELECT phrase, n, sum(numnode(phrase) + 1) OVER (ORDER BY n) - 1 AS nodes
                FROM (
                    SELECT phrase, min(w.n) AS n
                    FROM regexp_split_to_table(%(question)s, '[[:space:]]+') WITH ORDINALITY AS w(word, n),
                        LATERAL phraseto_tsquery('english', w.word) AS phrase
                    WHERE numnode(phrase) BETWEEN 1 AND %(max_phrase_nodes)s
                    GROUP BY phrase
                    ORDER BY n
                    LIMIT %(max_words)s
                ) words
            ) counted
            WHERE nodes <= %(max_nodes)s
        ) q
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
    check_model_name(conn, embedder.name)


def check_model_name(conn: psycopg.Connection, name: str) -> None:
    """Raise ModelMismatch unless the stored vectors were made by the model `name`."""
    stored = stored_model(conn)
    if stored is None:
        raise ModelMismatch("no embedded corpus: run recall ingest")
    if stored != name:
        raise ModelMismatch(
            f"the stored vectors are from {stored}, not {name}: search with the embedder that made them, or ingest"
            " again with this one"
        )


def full_text(conn: psycopg.Connection, question: str, k: int = DEFAULT_K) -> list[Hit]:
    """The k chunks whose search vectors best match any of the question's words, by ts_rank_cd."""
    _check_sizes(k, k)
    sql = f"WITH {_FULL_TEXT} SELECT {_HIT_COLUMNS}, f.score, f.rank, NULL FROM fts f {_JOIN} ORDER BY f.rank"
    return _hits(conn, sql, {"question": question, "n": k, **_TEXT_LIMITS}, ef_search=None)


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
    params = {
        "question": question,
        "vector": vector_literal(query_vector),
        "n": candidates,
        "k": k,
        "rrf_k": rrf_k,
        **_TEXT_LIMITS,
    }
    return _hits(conn, sql, params, ef_search=candidates)


def search(
    conn: psycopg.Connection,
    question: str,
    embedder: Embedder | None,
    mode: Mode = "hybrid",
    k: int = DEFAULT_K,
    *,
    candidates: int = DEFAULT_CANDIDATES,
    query_vector: Sequence[float] | None = None,
) -> list[Hit]:
    """Retrieve with one mode, embedding the question when the mode needs it (unless `query_vector` is the embedder's
    vector for it already: the API embeds before taking a connection). The model check and the query run in one
    transaction of their own, so the connection must not be inside a transaction already."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if mode == "full-text":
        return full_text(conn, question, k)
    if embedder is None:
        raise ValueError(f"{mode} retrieval needs an embedder")
    if conn.info.transaction_status != pq.TransactionStatus.IDLE:
        # Inside the caller's transaction, ours would be a savepoint, where SET TRANSACTION can't run.
        raise ValueError("search runs its own transaction: call it outside one")
    if query_vector is None:
        query_vector = embedder.embed_query(question)
    # One snapshot for the check and the query: in two, an ingest with another model committing between them would
    # pair this question's vector with that model's vectors, the mix the check exists to refuse.
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        check_model(conn, embedder)
        if mode == "vector":
            return vector(conn, query_vector, k)
        return hybrid(conn, question, query_vector, k, candidates=candidates)


_JOIN = "JOIN chunks c ON c.id = f.id JOIN documents d ON d.id = c.document_id"


def _check_sizes(k: int, candidates: int) -> None:
    if not (1 <= k <= MAX_K and 1 <= candidates <= MAX_K):
        raise ValueError(f"k and candidates must be from 1 to {MAX_K}, not {k} and {candidates}")


def _hits(conn: psycopg.Connection, sql: str, params: dict[str, Any], *, ef_search: int | None) -> list[Hit]:
    with conn.transaction():
        if ef_search is not None:
            # Both local to this transaction. An HNSW scan picks its ef_search nearest index entries and only then
            # drops the dead ones (rows an ingest updated or deleted stay in the index until VACUUM), so even with
            # ef_search at the list length the list came back short after an ingest that rewrote chunks: 31 of 60
            # after a model change. An iterative scan keeps searching, in distance order, until the LIMIT is met or
            # it has visited hnsw.max_scan_tuples entries (20,000), dead ones included; ingest vacuums after it
            # changes anything, so only rows deleted some other way can use that up.
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(ef_search, EF_SEARCH_MIN)),))
            conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
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
