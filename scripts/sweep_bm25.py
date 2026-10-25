"""BM25 across k1 and b on the labelled questions, with ts_rank_cd and the defaults for reference.

    .venv/bin/python scripts/sweep_bm25.py [--questions eval/questions.jsonl]

Needs `recall ingest` ($RECALL_DATABASE_URL); full-text search loads no model, so this takes seconds. As in
scripts/sweep_fusion.py, the questions are split in two by their place in the file (odd and even), so a setting
picked on one half can be checked on the other. Prints recall@1/5/10 and MRR@20 for each half and for all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sweep_fusion import row

from recall import db, retrieval
from recall.evaluation import DEFAULT_QUESTIONS, DEPTH, Ranked, gold_chunks, load_questions

K1S = (0.5, 0.9, 1.2, 1.5, 2.0)
BS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = a.parse_args()
    questions = load_questions(args.questions)
    with db.connect() as conn:
        gold = gold_chunks(conn, questions)

        def ranked(
            ranking: retrieval.Ranking, k1: float = retrieval.BM25_K1, b: float = retrieval.BM25_B
        ) -> list[Ranked]:
            return [
                Ranked(
                    q,
                    gold[q.id],
                    tuple(
                        h.chunk_id for h in retrieval.full_text(conn, q.question, DEPTH, ranking=ranking, k1=k1, b=b)
                    ),
                )
                for q in questions
            ]

        print(f"{len(questions)} questions; each cell: recall@1, recall@5, recall@10, MRR@{DEPTH}\n")
        print("| setting | odd places in the file | even places | all |\n|---|---|---|---|")
        print(row("ts_rank_cd", ranked("ts_rank_cd")))
        for k1 in K1S:
            for b in BS:
                default = " (default)" if (k1, b) == (retrieval.BM25_K1, retrieval.BM25_B) else ""
                print(row(f"bm25, k1 {k1}, b {b}{default}", ranked("bm25", k1, b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
