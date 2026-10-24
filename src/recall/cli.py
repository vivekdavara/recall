"""The recall command: fetch and check the corpus, chunk it, load it into PostgreSQL, search it, answer from it,
and measure retrieval on the labelled questions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import psycopg

from recall import chunking, db, retrieval
from recall.corpus import fetch, source
from recall.corpus.fetch import CorpusError
from recall.corpus.load import load
from recall.corpus.normalize import NormalizeError
from recall.embedding import DEFAULT_MODEL, Embedder, HashEmbedder, ModelUnavailable
from recall.evaluation import QuestionSetError


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


def question_problem(question: str) -> str | None:
    """Why PostgreSQL can't take this question, if it can't."""
    try:
        question.encode("utf-8")
    except UnicodeEncodeError:
        # Python reads undecodable argv bytes as lone surrogates (e.g. Latin-1 from a file), which psycopg can't send.
        return "the question isn't valid UTF-8"
    if "\x00" in question:
        return "the question holds a NUL character"  # PostgreSQL text can't hold one
    return None


def cmd_search(args: argparse.Namespace) -> int:
    problem = question_problem(args.question)
    if problem:
        print(f"recall: {problem}", file=sys.stderr)
        return 1
    try:
        embedder = None
        if args.mode != "full-text":
            # Check the stored vectors' model before loading one (seconds, or a download), on a connection closed
            # before the load, so no connection sits idle through it (a server's idle timeout would end it). search
            # checks again, with the query, in one snapshot. Full-text search needs no model.
            with db.connect() as conn:
                retrieval.check_model_name(conn, HashEmbedder().name if args.embedder == "hash" else DEFAULT_MODEL.name)
            embedder = make_embedder(args.embedder)
        with db.connect() as conn:
            hits = retrieval.search(conn, args.question, embedder, args.mode, args.k, candidates=args.candidates)
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        # Search doesn't migrate: nothing ingested yet, or a database from an older version (ingest migrates it).
        # Only here is that the likely cause; elsewhere such an error is a bug and keeps its traceback.
        url = db.describe_url(db.database_url())
        print(f"recall: the database {url} has no corpus or an older schema: run recall ingest", file=sys.stderr)
        return 1
    if not hits:
        print("no results")
    for i, h in enumerate(hits, 1):
        ranks = [f"{name} #{r}" for name, r in (("full-text", h.full_text_rank), ("vector", h.vector_rank)) if r]
        print(f"{i}. {h.path}{h.url[h.url.find('#') :] if '#' in h.url else ''}  {' > '.join(h.heading_path)}")
        print(f"   score {h.score:.4f} ({', '.join(ranks)}); chars {h.start}-{h.end}")
        text = " ".join(h.content.split())
        print(f"   {text[: args.width]}{'...' if len(text) > args.width else ''}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from recall.answer import ask
    from recall.llm import ProviderError, provider_from_environment
    from recall.rerank import make_reranker

    problem = question_problem(args.question)
    if problem:
        print(f"recall: {problem}", file=sys.stderr)
        return 1
    if not args.no_rerank and args.passages > args.depth:
        print(f"recall: passages must be from 1 to depth ({args.depth}), not {args.passages}", file=sys.stderr)
        return 1
    try:
        provider = provider_from_environment()
    except (ValueError, ProviderError) as e:  # RECALL_LLM names no provider, or the live one's package is missing
        print(f"recall: {e}", file=sys.stderr)
        return 1
    try:
        embedder = None
        if args.mode != "full-text":
            # As in search: check the stored vectors' model before loading one, on a connection closed meanwhile.
            with db.connect() as conn:
                retrieval.check_model_name(conn, HashEmbedder().name if args.embedder == "hash" else DEFAULT_MODEL.name)
            embedder = make_embedder(args.embedder)
        reranker = make_reranker(args.reranker)
        with db.connect() as conn:
            result, timings = ask(
                conn,
                args.question,
                embedder,
                reranker,
                provider,
                mode=args.mode,
                passages=args.passages,
                depth=args.depth,
                candidates=args.candidates,
                use_reranker=not args.no_rerank,
            )
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        url = db.describe_url(db.database_url())
        print(f"recall: the database {url} has no corpus or an older schema: run recall ingest", file=sys.stderr)
        return 1
    if args.json:
        from recall.api import _answer

        print(json.dumps(_answer(result, timings), indent=2))
        return 0
    if result.method == "none":
        print("no answer: the passages retrieved don't answer the question" if result.passages else "no results")
    else:
        print(result.text)
    if result.claims:
        print("\ncitations:")
        for n, h in enumerate(result.passages, 1):
            cited = [x for c in result.claims for x in c.citations if x.passage == n]
            if cited:
                print(f"  [{n}] {h.url}  ({h.path}, chars {', '.join(f'{x.start}-{x.end}' for x in cited)})")
    by = f"{result.provider} ({result.model})" if result.method == "llm" else result.method
    print(f"\nanswered by: {by}; {len(result.passages)} passages")
    for problem in result.problems:
        print(f"  note: {problem}")
    print(
        f"time: retrieve {timings.retrieve:.1f} ms, rerank {timings.rerank:.1f} ms, answer {timings.answer:.1f} ms"
        " (models loaded before)"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from recall import evaluation
    from recall.answer import DEFAULT_PASSAGES
    from recall.rerank import make_reranker

    try:
        questions = evaluation.load_questions(args.questions)
        baseline = json.loads(args.check.read_text(encoding="utf-8")) if args.check else None
    except (OSError, ValueError) as e:  # QuestionSetError is a ValueError, and so is bad JSON in the baseline
        print(f"recall: {e}", file=sys.stderr)
        return 1
    provider = None
    provider_name = "no LLM"
    if args.answers:
        from recall.llm import ProviderError, provider_from_environment

        try:
            provider = provider_from_environment()
        except (ValueError, ProviderError) as e:
            print(f"recall: {e}", file=sys.stderr)
            return 1
        provider_name = f"provider {provider.name}" if provider is not None else "no LLM"
    systems = args.systems
    embedder = None
    if args.answers or any(s != "full-text" for s in systems):  # answers are hybrid, as ask's are
        # As in search: check the stored vectors' model before loading one.
        with db.connect() as conn:
            retrieval.check_model_name(conn, HashEmbedder().name if args.embedder == "hash" else DEFAULT_MODEL.name)
        embedder = make_embedder(args.embedder)
    reranker = make_reranker(args.reranker) if args.answers or any(s in evaluation.RERANKED for s in systems) else None
    start = time.perf_counter()
    with db.connect() as conn:
        result = evaluation.evaluate(
            conn,
            questions,
            evaluation.digest(args.questions),
            embedder,
            reranker,
            systems,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
        )
        answers = None
        if args.answers:
            from recall.faithfulness import answer_all, score_answers

            assert reranker is not None
            gold = evaluation.gold_chunks(conn, questions)
            answers = score_answers(answer_all(conn, questions, gold, embedder, reranker, provider))
    seconds = time.perf_counter() - start
    print(f"{len(questions)} questions ({args.questions}), {seconds:.1f} s\n")
    print(evaluation.table(result.scores))
    for kind in evaluation.KINDS:
        print(f"\n{kind} questions:\n")
        print(evaluation.table({s: kinds[kind] for s, kinds in result.by_kind.items()}))
    against = "hybrid+rerank" if "hybrid+rerank" in result.ranks else systems[0]
    if len(result.ranks) > 1:
        print(f"\nquestions found only by each system / only by {against}, and the sign test's p:\n")
        print(evaluation.paired_table(result.ranks, against))
    body = result.to_json()
    if answers is not None:
        body["answers"] = answers.to_json()
        print(f"\nanswers (ask's defaults: hybrid, reranked, {DEFAULT_PASSAGES} passages; {provider_name}):\n")
        print(
            "| answers | methods | claims | supported claims | all claims supported | cites gold | gold in passages |\n"
            "|---:|---|---:|---:|---:|---:|---:|\n"
            f"| {answers.answers} | {', '.join(f'{k} {v}' for k, v in sorted(answers.methods.items()))}"
            f" | {answers.claims} | {answers.supported_claims:.3f} | {answers.fully_supported:.3f}"
            f" | {answers.cites_gold:.3f} | {answers.gold_in_passages:.3f} |"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.check:
        # Even a baseline file holding JSON null goes to compare, which refuses it: a gate that can't compare fails.
        problems = evaluation.compare(body, baseline)
        if problems:
            print(f"\nworse than {args.check}:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"\nno worse than {args.check}")
    return 0


def systems_list(text: str) -> list[str]:
    from recall.evaluation import SYSTEMS

    names = [s.strip() for s in text.split(",") if s.strip()]
    unknown = [s for s in names if s not in SYSTEMS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"use a comma-separated list of {', '.join(SYSTEMS)}")
    return list(dict.fromkeys(names))


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("recall: the API needs FastAPI and uvicorn: pip install -e '.[api]'", file=sys.stderr)
        return 1
    from recall.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
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
    a = sub.add_parser("ask", help="answer a question with citations (the LLM by $RECALL_LLM, else extractive)")
    a.add_argument("question")
    a.add_argument("--mode", choices=retrieval.MODES, default="hybrid")
    a.add_argument(
        "--passages", type=list_length, default=5, help="how many passages to answer from (default %(default)s)"
    )
    a.add_argument(
        "--depth", type=list_length, default=20, help="how many retrieved chunks to rerank (default %(default)s)"
    )
    a.add_argument("--candidates", type=list_length, default=retrieval.DEFAULT_CANDIDATES)
    a.add_argument("--no-rerank", action="store_true", help="answer from the retrieval order")
    a.add_argument("--json", action="store_true", help="print what POST /ask returns")
    add_embedder_option(a)
    a.add_argument(
        "--reranker",
        choices=("model", "overlap"),
        default=os.environ.get("RECALL_RERANKER", "model"),
        help="model: the pinned cross-encoder (default; $RECALL_RERANKER overrides); overlap: the stub",
    )
    a.set_defaults(run=cmd_ask)
    e = sub.add_parser("eval", help="measure recall@k and MRR of each retrieval system on the labelled questions")
    e.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"), help="(default %(default)s)")
    e.add_argument(
        "--systems",
        type=systems_list,
        default=["full-text", "vector", "hybrid", "vector+rerank", "hybrid+rerank"],
        help="comma-separated (default: all five: full-text, vector, hybrid, vector+rerank, hybrid+rerank)",
    )
    e.add_argument("--candidates", type=list_length, default=retrieval.DEFAULT_CANDIDATES, help="hybrid's lists")
    e.add_argument("--rrf-k", type=int, default=retrieval.RRF_K, help="the k in 1 / (k + rank) (default %(default)s)")
    e.add_argument("--json", type=Path, help="write the report (scores, settings, every rank) here")
    e.add_argument("--check", type=Path, help="exit 1 if worse than this baseline (a report written by --json)")
    e.add_argument(
        "--answers",
        action="store_true",
        help="also answer every question as recall ask does ($RECALL_LLM) and check the answers' faithfulness",
    )
    add_embedder_option(e)
    e.add_argument(
        "--reranker",
        choices=("model", "overlap"),
        default=os.environ.get("RECALL_RERANKER", "model"),
        help="model: the pinned cross-encoder (default; $RECALL_RERANKER overrides); overlap: the stub",
    )
    e.set_defaults(run=cmd_eval)
    serve = sub.add_parser("serve", help="run the HTTP API (POST /ask, POST /search, GET /healthz)")
    serve.add_argument("--host", default="127.0.0.1", help="the address to listen on (default %(default)s)")
    serve.add_argument("--port", type=positive, default=8109, help="(default %(default)s)")
    serve.set_defaults(run=cmd_serve)
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
    if getattr(args, "reranker", "model") not in ("model", "overlap"):
        p.error(f"RECALL_RERANKER must be one of model, overlap, not {args.reranker!r}")
    try:
        code: int = args.run(args)
    except (CorpusError, NormalizeError, ModelUnavailable, retrieval.ModelMismatch, QuestionSetError) as e:
        print(f"recall: {e}", file=sys.stderr)
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
