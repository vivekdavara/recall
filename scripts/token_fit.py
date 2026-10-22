"""Do the corpus's chunks fit a model's input? Counts every passage (heading path + chunk text) in model tokens.

    .venv/bin/python scripts/token_fit.py [--max-tokens 256] [--model REPO@REVISION --max-seq N]

Prints the distribution of model tokens per passage, the ratio to the chunker's own count, and how many passages
the model would truncate. Needs the fetched corpus (recall fetch) and the `embed` extra; downloads only the
tokenizer.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--model", default=DEFAULT_MODEL.name, help="REPO@REVISION")
    p.add_argument("--max-seq", type=int, default=DEFAULT_MODEL.max_seq_length)
    p.add_argument("--data", default="data/corpus")
    args = p.parse_args()

    repo, _, revision = args.model.partition("@")
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision or None)
    documents = load(Path(args.data), fetch.load_manifest())
    counts: list[int] = []
    ratios: list[float] = []
    for doc in documents:
        for c in chunk_document(doc.content, args.max_tokens):
            n = len(tokenizer(passage_text(c.heading_path, c.text), truncation=False)["input_ids"])
            counts.append(n)
            ratios.append(n / c.token_count)
    counts.sort()
    ratios.sort()
    over = sum(1 for n in counts if n > args.max_seq)
    print(f"model {args.model}, max_seq_length {args.max_seq}; chunker max_tokens {args.max_tokens}")
    print(f"passages {len(counts)}; model tokens min {counts[0]} p50 {percentile(counts, 50)}", end="")
    print(f" p95 {percentile(counts, 95)} p99 {percentile(counts, 99)} max {counts[-1]}")
    print(f"model tokens per chunker token: p50 {ratios[len(ratios) // 2]:.2f} max {ratios[-1]:.2f}")
    print(f"truncated (over {args.max_seq}): {over} ({100 * over / len(counts):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
