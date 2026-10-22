"""Measure retrieval on the ingested real corpus: HNSW against exact search, and each step's latency.

    .venv/bin/python scripts/measure_retrieval.py [--questions 300] [--warmup 30] [-k 10]

Needs `recall ingest` with the real model ($RECALL_DATABASE_URL) and the `embed` extra.

The questions are synthetic: the corpus's distinct section titles (a heading path's last title), shuffled with a fixed
seed. Each title is also in the text embedded for its own section's chunks, so these are easy questions: fine for
timing and for comparing HNSW with exact search, not a measure of answer quality (day 4's labelled set is).

1. HNSW against exact: for each question, the vector top k through the HNSW index at several hnsw.ef_search values
   (retrieval uses max(k, 100); pgvector's default is 40) and by a sequential scan (index scans off, so every
   distance is computed; the plans are checked); recall@k is the share of the exact top k that HNSW returned.
2. Latency: embedding the question, each retriever's query, and hybrid end to end (embed + query), each over every
   question after an untimed warm-up of the first `--warmup` questions, as nearest-rank percentiles.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Callable

import psycopg

from recall import db, retrieval
from recall.embedding import SentenceTransformerEmbedder

EF_SEARCH = (10, 20, 40, 60, 100)


def questions(conn: psycopg.Connection, n: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT heading_path[cardinality(heading_path)] FROM chunks WHERE cardinality(heading_path) > 0"
    ).fetchall()
    titles = sorted(r[0] for r in rows)
    random.Random(2026).shuffle(titles)
    return titles[:n]


_KNN = "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s"


def _settings(conn: psycopg.Connection, exact: bool, ef_search: int) -> None:
    # The planner picks a sequential scan over the index once ef_search is high enough for this table (about 80 to
    # 100 at 1,503 rows), so each side forces its own plan; check_plan verifies it.
    conn.execute("SET LOCAL enable_indexscan = off" if exact else "SET LOCAL enable_seqscan = off")
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))


def nearest(conn: psycopg.Connection, vector: str, k: int, *, exact: bool, ef_search: int = 40) -> list[int]:
    with conn.transaction():
        _settings(conn, exact, ef_search)
        rows = conn.execute(_KNN, (vector, k)).fetchall()
    return [r[0] for r in rows]


def plan_of(conn: psycopg.Connection, sql: str, params: tuple[object, ...], setup: Callable[[], object]) -> str:
    with conn.transaction():
        setup()
        plan = "\n".join(r[0] for r in conn.execute(("EXPLAIN " + sql).encode(), params))
    return "HNSW index" if "chunks_embedding_hnsw" in plan else "sequential scan" if "Seq Scan" in plan else plan


def check_plan(conn: psycopg.Connection, vector: str, k: int, *, exact: bool, ef_search: int) -> None:
    """Fail unless the exact query really scans and the HNSW one really uses the index."""
    got = plan_of(conn, _KNN, (vector, k), lambda: _settings(conn, exact, ef_search))
    if got != ("sequential scan" if exact else "HNSW index"):
        sys.exit(f"the {'exact' if exact else 'HNSW'} query at ef_search {ef_search} has the plan: {got}")


def timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def ms(values: list[float]) -> str:
    """p50, p95, p99 and max of values, by nearest rank (rank ceil(n * q / 100), in integers)."""
    v = sorted(values)
    p = {q: v[max(1, -(-len(v) * q // 100)) - 1] for q in (50, 95, 99)}
    return f"{p[50]:.2f} | {p[95]:.2f} | {p[99]:.2f} | {v[-1]:.2f}"


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--questions", type=int, default=300)
    a.add_argument("--warmup", type=int, default=30)
    a.add_argument("-k", type=int, default=10)
    args = a.parse_args()

    embedder = SentenceTransformerEmbedder()
    with db.connect(application_name="recall-measure") as conn:
        retrieval.check_model(conn, embedder)
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()
        qs = questions(conn, args.questions + args.warmup)
        warm, qs = qs[: args.warmup], qs[args.warmup :]
        print(f"{chunks[0] if chunks else 0} chunks; {len(qs)} questions (+{len(warm)} warm-up); k = {args.k}\n")

        floats = {q: embedder.embed_query(q) for q in warm + qs}
        vectors = {q: db.vector_literal(v) for q, v in floats.items()}
        print("| ef_search | HNSW recall@k vs exact (mean) | questions with all k |")
        print("|---|---|---|")
        check_plan(conn, vectors[qs[0]], args.k, exact=True, ef_search=40)
        exact = {q: nearest(conn, vectors[q], args.k, exact=True) for q in qs}
        for ef in EF_SEARCH:
            check_plan(conn, vectors[qs[0]], args.k, exact=False, ef_search=ef)
            hits = [len(set(nearest(conn, vectors[q], args.k, exact=False, ef_search=ef)) & set(exact[q])) for q in qs]
            full = sum(1 for h in hits if h == args.k)
            print(f"| {ef} | {sum(hits) / (args.k * len(qs)):.4f} | {full} of {len(qs)} |")

        steps: dict[str, Callable[[str], object]] = {
            "embed the question": lambda q: embedder.embed_query(q),
            "full-text query": lambda q: retrieval.full_text(conn, q, args.k),
            "vector query": lambda q: retrieval.vector(conn, floats[q], args.k),
            "hybrid query": lambda q: retrieval.hybrid(conn, q, floats[q], args.k),
            "hybrid end to end": lambda q: retrieval.search(conn, q, embedder, "hybrid", args.k),
        }
        # Which plan retrieval.vector's query gets, as it sets things (ef_search max(k, floor), iterative scan).
        floor = retrieval.EF_SEARCH_MIN
        for n in sorted({args.k, retrieval.DEFAULT_CANDIDATES}):

            def as_retrieval(n: int = n) -> None:
                conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(n, floor)),))
                conn.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")

            print(f"\nretrieval's vector query for {n} rows (ef_search {max(n, floor)}) uses: ", end="")
            print(plan_of(conn, _KNN, (vectors[qs[0]], n), as_retrieval))
        print("\n| step | p50 ms | p95 ms | p99 ms | max ms |")
        print("|---|---|---|---|---|")
        for name, step in steps.items():
            for q in warm:
                step(q)
            print(f"| {name} | {ms([timed(lambda q=q, step=step: step(q)) for q in qs])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
