"""The recall command: fetch and check the corpus, chunk it, load it into PostgreSQL, and search it."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg

from recall import chunking, db, retrieval
from recall.corpus import fetch, source
from recall.corpus.fetch import CorpusError
from recall.corpus.load import load
from recall.corpus.normalize import NormalizeError
from recall.embedding import Embedder, HashEmbedder, ModelUnavailable


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


def cmd_show(args: argparse.Namespace) -> int:
    documents = {d.path: d for d in load(corpus_dir(), fetch.load_manifest())}
    doc = documents.get(args.page)
    if doc is None:
        print(
            f"recall: no page {args.page!r} in the corpus (pages look like tutorial/query-params.md)", file=sys.stderr
        )
        return 1
    print(f"{doc.path}: {doc.title} <{doc.url}>")
    for c in chunking.chunk_document(doc.content, args.max_tokens):
        where = " > ".join(c.heading_path) or "(before the first heading)"
        anchor = f" #{c.anchor}" if c.anchor else ""
        print(f"\n--- chunk {c.ordinal}: {where}{anchor}; [{c.start}:{c.end}], {c.token_count} tokens")
        print(c.text)
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    with db.connect() as conn:
        applied = db.migrate(conn)
    print(f"applied {', '.join(applied)}" if applied else "nothing to apply")
    return 0


EMBEDDERS = ("model", "hash")


def make_embedder(kind: str) -> Embedder:
    """`model` is the pinned sentence-transformers model (the `embed` extra); `hash` the dependency-free stub."""
    if kind == "hash":
        return HashEmbedder()
    if kind != "model":
        raise ValueError(f"unknown embedder {kind!r}: use one of {', '.join(EMBEDDERS)}")
    from recall.embedding import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder()


def cmd_ingest(args: argparse.Namespace) -> int:
    from recall.ingest import ingest

    manifest = fetch.load_manifest()
    documents = load(corpus_dir(), manifest)
    embedder = make_embedder(args.embedder)
    with db.connect() as conn:
        db.migrate(conn)
        report = ingest(conn, documents, source_commit=manifest.commit, max_tokens=args.max_tokens, embedder=embedder)
    print(
        f"documents: {report.added} added, {report.updated} updated, {report.unchanged} unchanged,"
        f" {report.removed} removed; {report.chunks_written} chunks written, {report.chunks_embedded} embedded"
        f" ({embedder.name})"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        args.question.encode("utf-8")
    except UnicodeEncodeError:
        # Python reads undecodable argv bytes as lone surrogates (e.g. Latin-1 from a file), which psycopg can't send.
        print("recall: the question isn't valid UTF-8", file=sys.stderr)
        return 1
    # Full-text search needs no model: don't load one (it takes seconds).
    embedder = None if args.mode == "full-text" else make_embedder(args.embedder)
    with db.connect() as conn:
        hits = retrieval.search(conn, args.question, embedder, args.mode, args.k, candidates=args.candidates)
    if not hits:
        print("no results")
    for i, h in enumerate(hits, 1):
        ranks = [f"{name} #{r}" for name, r in (("full-text", h.full_text_rank), ("vector", h.vector_rank)) if r]
        print(f"{i}. {h.path}{h.url[h.url.find('#') :] if '#' in h.url else ''}  {' > '.join(h.heading_path)}")
        print(f"   score {h.score:.4f} ({', '.join(ranks)}); chars {h.start}-{h.end}")
        text = " ".join(h.content.split())
        print(f"   {text[: args.width]}{'...' if len(text) > args.width else ''}")
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
        if name == "ingest":
            add_embedder_option(c)
        c.set_defaults(run=run)
    show = sub.add_parser("show", help="print one page's chunks, with their spans and heading paths")
    show.add_argument("page", help="a page path, e.g. tutorial/query-params.md")
    show.add_argument("--max-tokens", type=positive, default=chunking.DEFAULT_MAX_TOKENS)
    show.set_defaults(run=cmd_show)
    sub.add_parser("migrate", help="apply the schema migrations").set_defaults(run=cmd_migrate)
    q = sub.add_parser("search", help="retrieve the chunks that best match a question")
    q.add_argument("question")
    q.add_argument("--mode", choices=retrieval.MODES, default="hybrid")
    q.add_argument("-k", type=list_length, default=retrieval.DEFAULT_K, help="how many chunks (default %(default)s)")
    q.add_argument(
        "--candidates",
        type=list_length,
        default=retrieval.DEFAULT_CANDIDATES,
        help="hybrid: how many chunks each retriever hands to the fusion (default %(default)s)",
    )
    q.add_argument("--width", type=positive, default=160, help="characters of each chunk to print")
    add_embedder_option(q)
    q.set_defaults(run=cmd_search)
    return p


def add_embedder_option(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--embedder",
        choices=EMBEDDERS,
        default=os.environ.get("RECALL_EMBEDDER", "model"),
        help="model: the pinned sentence-transformers model (default; $RECALL_EMBEDDER overrides); hash: the stub",
    )


def list_length(text: str) -> int:
    value = positive(text)
    if value > retrieval.MAX_K:
        raise argparse.ArgumentTypeError(f"must be at most {retrieval.MAX_K}, not {value}")
    return value


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if getattr(args, "embedder", "model") not in EMBEDDERS:
        # argparse checks choices on the command line, not on a default taken from $RECALL_EMBEDDER.
        p.error(f"RECALL_EMBEDDER must be one of {', '.join(EMBEDDERS)}, not {args.embedder!r}")
    try:
        code: int = args.run(args)
    except (CorpusError, NormalizeError, ModelUnavailable, retrieval.ModelMismatch) as e:
        print(f"recall: {e}", file=sys.stderr)
        return 1
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        # Searching before anything was ingested, or a database from an older version (ingest migrates it): say what
        # to run, not a traceback.
        print(
            f"recall: the database {db.describe_url(db.database_url())} has no corpus or an older schema:"
            " run recall ingest",
            file=sys.stderr,
        )
        return 1
    except (psycopg.OperationalError, db.BadDatabaseURL) as e:
        # One line (psycopg's message has one per address it tried), and nothing of the password.
        url = db.database_url()
        message = f"recall: can't use the database {db.describe_url(url)}: {db.describe_error(url, e)}"
        print(" ".join(message.split()), file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
