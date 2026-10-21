"""The recall command: fetch and check the corpus, chunk it, and load it into PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from recall import chunking, db
from recall.corpus import fetch, source
from recall.corpus.fetch import CorpusError
from recall.corpus.load import load
from recall.corpus.normalize import NormalizeError


def data_dir() -> Path:
    return Path(os.environ.get("RECALL_DATA_DIR", "data"))


def corpus_dir() -> Path:
    return data_dir() / "corpus"


def percentile(sorted_values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile of an ascending list (q in 0..100)."""
    if not sorted_values:
        raise ValueError("no values")
    rank = max(1, -(-len(sorted_values) * q // 100))
    return sorted_values[int(rank) - 1]


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = fetch.load_manifest()
    changed = fetch.fetch(corpus_dir(), manifest, args.url)
    state = "fetched" if changed else "already up to date:"
    print(f"{state} {len(manifest.files)} files of {manifest.repository} {manifest.tag} ({manifest.commit[:12]})")
    print(f"in {corpus_dir()}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    problems = fetch.verify(corpus_dir(), fetch.load_manifest())
    for p in problems:
        print(p)
    if problems:
        print(f"{len(problems)} problems; run recall fetch", file=sys.stderr)
        return 1
    print(f"{corpus_dir()} matches the manifest")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    manifest = fetch.build_manifest(Path(args.archive), commit=source.COMMIT, tag=source.TAG)
    text = manifest.to_json()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(f"{len(manifest.files)} files, {len(manifest.pages())} pages", file=sys.stderr)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    documents = load(corpus_dir(), fetch.load_manifest())
    counts: list[int] = []
    for doc in documents:
        counts.extend(c.token_count for c in chunking.chunk_document(doc.content, args.max_tokens))
    counts.sort()
    chars = sum(len(d.content) for d in documents)
    print(f"documents       {len(documents)}")
    print(f"characters      {chars}")
    print(f"chunks          {len(counts)} (max_tokens {args.max_tokens})")
    print(f"tokens          {sum(counts)}")
    p50, p95 = percentile(counts, 50), percentile(counts, 95)
    print(f"tokens a chunk  min {counts[0]} / p50 {p50} / p95 {p95} / max {counts[-1]}")
    print(f"chunks at max   {sum(1 for n in counts if n == args.max_tokens)}")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    with db.connect() as conn:
        applied = db.migrate(conn)
    print(f"applied {', '.join(applied)}" if applied else "nothing to apply")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from recall.ingest import ingest

    manifest = fetch.load_manifest()
    documents = load(corpus_dir(), manifest)
    with db.connect() as conn:
        db.migrate(conn)
        report = ingest(conn, documents, source_commit=manifest.commit, max_tokens=args.max_tokens)
    print(
        f"documents: {report.added} added, {report.updated} updated, {report.unchanged} unchanged,"
        f" {report.removed} removed; {report.chunks_written} chunks written"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recall", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download the pinned corpus into $RECALL_DATA_DIR/corpus (default data/)")
    f.add_argument("--url", default=source.ARCHIVE_URL, help="the archive to download (default: the pinned commit)")
    f.set_defaults(run=cmd_fetch)
    sub.add_parser("verify", help="check the extracted corpus against the manifest").set_defaults(run=cmd_verify)
    m = sub.add_parser("manifest", help="build a manifest from an archive (when the pinned version changes)")
    m.add_argument("archive")
    m.add_argument("--out", help="write here instead of standard output")
    m.set_defaults(run=cmd_manifest)
    for name, run, text in [
        ("stats", cmd_stats, "chunk the corpus and print its sizes"),
        ("ingest", cmd_ingest, "load the corpus into PostgreSQL ($RECALL_DATABASE_URL)"),
    ]:
        c = sub.add_parser(name, help=text)
        c.add_argument("--max-tokens", type=positive, default=chunking.DEFAULT_MAX_TOKENS)
        c.set_defaults(run=run)
    sub.add_parser("migrate", help="apply the schema migrations").set_defaults(run=cmd_migrate)
    return p


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        code: int = args.run(args)
    except (CorpusError, NormalizeError) as e:
        print(f"recall: {e}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
