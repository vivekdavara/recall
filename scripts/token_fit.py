"""Do the corpus's chunks fit a model's input? Counts every passage (heading path + chunk text) in model tokens.

    .venv/bin/python scripts/token_fit.py [--max-tokens 256] [--model REPO@REVISION --max-seq N]
    .venv/bin/python scripts/token_fit.py --reranker --question "How do I upload a file?"

Prints the distribution of model tokens per passage, the ratio to the chunker's own count, and how many passages
the model would truncate. With --question, each passage is counted as a pair with that question, as a
cross-encoder reads it ([CLS] question [SEP] passage [SEP]); --reranker picks the pinned reranker and its length.
Needs the fetched corpus (recall fetch) and the `embed` extra; downloads only the tokenizer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

from recall.chunking import DEFAULT_MAX_TOKENS, chunk_document
from recall.cli import percentile
from recall.corpus import fetch
from recall.corpus.load import load
from recall.embedding import DEFAULT_MODEL, passage_text
from recall.rerank import DEFAULT_RERANKER


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--model", default=DEFAULT_MODEL.name, help="REPO@REVISION")
    p.add_argument("--max-seq", type=int, default=DEFAULT_MODEL.max_seq_length)
    p.add_argument("--data", default="data/corpus")
    p.add_argument("--question", help="count each passage as a pair with this question")
    p.add_argument("--reranker", action="store_true", help="the pinned reranker's tokenizer and max_length")
    args = p.parse_args()
    if args.reranker:
        args.model, args.max_seq = DEFAULT_RERANKER.name, DEFAULT_RERANKER.max_length

    repo, _, revision = args.model.partition("@")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision or None)
    documents = load(Path(args.data), fetch.load_manifest())
    counts: list[int] = []
    ratios: list[float] = []
    for doc in documents:
        for c in chunk_document(doc.content, args.max_tokens):
            passage = passage_text(c.heading_path, c.text)
            if args.question is None:
                n = len(tokenizer(passage, truncation=False)["input_ids"])
            else:
                n = len(tokenizer(args.question, passage, truncation=False)["input_ids"])
            counts.append(n)
            ratios.append(n / c.token_count)
    counts.sort()
    ratios.sort()
    over = sum(1 for n in counts if n > args.max_seq)
    print(f"model {args.model}, max_seq_length {args.max_seq}; chunker max_tokens {args.max_tokens}")
    if args.question is not None:
        print(f"pairs with the question {args.question!r}")
    print(f"passages {len(counts)}; model tokens min {counts[0]} p50 {percentile(counts, 50)}", end="")
    print(f" p95 {percentile(counts, 95)} p99 {percentile(counts, 99)} max {counts[-1]}")
    print(f"model tokens per chunker token: p50 {ratios[len(ratios) // 2]:.2f} max {ratios[-1]:.2f}")
    print(f"truncated (over {args.max_seq}): {over} ({100 * over / len(counts):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
