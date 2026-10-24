"""Hybrid retrieval across RRF k and candidates on the labelled questions, with vector alone for reference.

    .venv/bin/python scripts/sweep_fusion.py [--questions eval/questions.jsonl]

Needs `recall ingest` with the real model ($RECALL_DATABASE_URL) and the `embed` extra. Each question is embedded once;
then hybrid's top 20 is taken at every (candidates, RRF k) pair. Choosing a setting by its score on the same questions
it is reported on overfits them, so the questions are split in two by their place in the file (odd and even: with
q001, q002, ... the odd and even ids): pick on one half, and the other half says whether the choice holds. Prints
recall@1/5/10 and MRR@20 for each half and for all.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from recall import db, retrieval
from recall.embedding import SentenceTransformerEmbedder
from recall.evaluation import DEFAULT_QUESTIONS, DEPTH, Ranked, gold_chunks, load_questions, score

CANDIDATES = (20, 50, 100)
RRF_KS = (1, 10, 30, 60, 120)


def half(ranked: Sequence[Ranked], parity: int) -> list[Ranked]:
    # By position in the file (first, third, ... and second, fourth, ...), whatever the ids look like.
    return [r for n, r in enumerate(ranked, 1) if n % 2 == parity]


def row(name: str, ranked: Sequence[Ranked]) -> str:
    cells = []
    for part in (half(ranked, 1), half(ranked, 0), list(ranked)):
        s = score(part)
        cells.append(f"{s.recall[1]:.3f} {s.recall[5]:.3f} {s.recall[10]:.3f} {s.mrr:.3f}")
    return f"| {name} | " + " | ".join(cells) + " |"


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = a.parse_args()
    questions = load_questions(args.questions)
    embedder = SentenceTransformerEmbedder()
    with db.connect() as conn:
        retrieval.check_model(conn, embedder)
        gold = gold_chunks(conn, questions)
        vectors = [embedder.embed_query(q.question) for q in questions]
        print(f"{len(questions)} questions; each cell: recall@1, recall@5, recall@10, MRR@{DEPTH}\n")
        print("| setting | odd places in the file | even places | all |\n|---|---|---|---|")
        ranked = [
            Ranked(q, gold[q.id], tuple(h.chunk_id for h in retrieval.vector(conn, v, DEPTH)))
            for q, v in zip(questions, vectors, strict=True)
        ]
        print(row("vector alone", ranked))
        for candidates in CANDIDATES:
            for rrf_k in RRF_KS:
                ranked = [
                    Ranked(
                        q,
                        gold[q.id],
                        tuple(
                            h.chunk_id
                            for h in retrieval.hybrid(conn, q.question, v, DEPTH, candidates=candidates, rrf_k=rrf_k)
                        ),
                    )
                    for q, v in zip(questions, vectors, strict=True)
                ]
                default = " (default)" if (candidates, rrf_k) == (retrieval.DEFAULT_CANDIDATES, retrieval.RRF_K) else ""
                print(row(f"hybrid, candidates {candidates}, k {rrf_k}{default}", ranked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
