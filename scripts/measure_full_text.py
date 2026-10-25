"""Time full-text search with each ranking on the labelled questions: ts_rank_cd against BM25.

    .venv/bin/python scripts/measure_full_text.py [--questions eval/questions.jsonl] [--k 20] [--rounds 2]

Needs `recall ingest` ($RECALL_DATABASE_URL); no model. Each round runs every question once with each ranking, the
rankings alternating by round so neither always runs on a warmer cache; prints each round's p50, p95 and max of the
query's wall time as the client sees it (one connection, the database on this machine).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from recall import db, retrieval
from recall.cli import percentile
from recall.evaluation import DEFAULT_QUESTIONS, load_questions


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    a.add_argument("--k", type=int, default=20)
    a.add_argument("--rounds", type=int, default=2)
    args = a.parse_args()
    questions = [q.question for q in load_questions(args.questions)]
    rankings: tuple[retrieval.Ranking, ...] = ("ts_rank_cd", "bm25")
    print(f"{len(questions)} questions, top {args.k}; milliseconds\n")
    print("| round | ranking | p50 | p95 | max |\n|---:|---|---:|---:|---:|")
    with db.connect() as conn:
        for r in range(1, args.rounds + 1):
            for ranking in rankings if r % 2 else rankings[::-1]:
                micros = []
                for q in questions:
                    t = time.perf_counter_ns()
                    retrieval.full_text(conn, q, args.k, ranking=ranking)
                    micros.append((time.perf_counter_ns() - t) // 1000)
                micros.sort()
                cells = [percentile(micros, p) / 1000 for p in (50, 95, 100)]
                print(f"| {r} | {ranking} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
