"""Review d2 findings for `recall search` and `recall ingest --embedder`. Each test fails on commit 2fba151."""

from __future__ import annotations

import psycopg
import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_cli import setup  # noqa: F401  (the fixture)

from recall import cli, db, embedding
from recall.corpus.fetch import Manifest


def use_schema(monkeypatch: pytest.MonkeyPatch, schema: str) -> None:
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")


def day_one_database(schema: str) -> None:
    """The schema as day 1's `recall migrate` left it: migration 001 only."""
    with connect_to(schema) as conn:
        version, sql = db.migrations()[0]
        conn.execute("CREATE TABLE schema_migrations (version text PRIMARY KEY, applied_at timestamptz DEFAULT now())")
        conn.execute(sql.encode())
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))


@pytest.mark.parametrize("argv", [["--mode", "full-text"], ["--mode", "hybrid", "--embedder", "hash"]])
def test_search_on_a_day_one_database_says_what_to_run(
    schema: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
):
    """A database migrated on day 1 has chunks but no search_vector, embedding or embedding_model column. `recall
    search` doesn't migrate, and main() turns only UndefinedTable into "has no corpus: run recall ingest"; here the
    query fails with psycopg.errors.UndefinedColumn, which escapes main() as a traceback. Correct: exit 1 with a
    one-line message that says to run recall migrate (or ingest)."""
    day_one_database(schema)
    use_schema(monkeypatch, schema)
    try:
        code = cli.main(["search", "x", *argv])
    except psycopg.errors.UndefinedColumn as e:
        pytest.fail(f"traceback instead of a message: {e!r}")
    assert code == 1
    assert "recall " in capsys.readouterr().err


def test_an_invalid_recall_embedder_is_refused_not_read_as_model(monkeypatch: pytest.MonkeyPatch):
    """--embedder has choices ["model", "hash"], but its default comes from $RECALL_EMBEDDER and argparse doesn't
    check a default against the choices. RECALL_EMBEDDER=Hash (or 'hsah', or '') is accepted, and make_embedder
    treats anything but "hash" as "model": the run silently loads the sentence-transformers model (seconds, or a
    download) and, for ingest, re-embeds the whole corpus with it. Correct: refuse a value that isn't a choice
    (argparse error, exit 2, or a ValueError from make_embedder)."""
    monkeypatch.setenv("RECALL_EMBEDDER", "Hash")
    loaded: list[str] = []
    monkeypatch.setattr(embedding, "SentenceTransformerEmbedder", lambda *a, **k: loaded.append("model"))
    try:
        args = cli.parser().parse_args(["search", "x"])
    except SystemExit as e:
        assert e.code == 2
        return
    try:
        cli.make_embedder(args.embedder)
    except ValueError:
        return
    pytest.fail(f"--embedder defaulted to {args.embedder!r} and loaded {loaded}")


def test_a_model_that_cannot_be_loaded_is_reported_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """ModelUnavailable's docstring: "sentence-transformers isn't installed, or the model can't be loaded". But
    SentenceTransformerEmbedder only turns an ImportError (and a wrong dimension) into it: when the model can't be
    loaded (offline with the pinned revision not cached, a corrupt cache, a revoked revision) SentenceTransformer
    raises OSError, which escapes main() as a traceback. Correct: ModelUnavailable, so `recall search`/`ingest`
    print "recall: ..." and exit 1. No database is touched: the model is loaded before connecting."""
    st = pytest.importorskip("sentence_transformers")

    def unloadable(*args: object, **kwargs: object) -> object:
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them")

    monkeypatch.setattr(st, "SentenceTransformer", unloadable)
    monkeypatch.setenv("RECALL_DATABASE_URL", "postgresql://127.0.0.1:1/never-used")
    try:
        code = cli.main(["search", "x", "--embedder", "model"])
    except OSError as e:
        pytest.fail(f"traceback instead of a message: {e!r}")
    assert code == 1
    # (Adjusted with the fix: torch prints a warning of its own to stderr when it is first imported; ours is last.)
    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith("recall: can't load BAAI/bge-small-en-v1.5@")


def test_a_question_that_is_not_utf8_is_reported_not_a_traceback(
    setup: tuple[str, Manifest],  # noqa: F811
    schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Python decodes argv with surrogateescape, so `recall search "$(printf 'caf\\xe9')"` (Latin-1 bytes from a
    file or an old terminal) gives the question 'caf\\udce9'. psycopg can't encode a lone surrogate and raises
    UnicodeEncodeError, which escapes main() as a traceback. Correct: exit 1 with a one-line message (or search
    with the undecodable bytes replaced)."""
    url, _ = setup
    use_schema(monkeypatch, schema)
    assert cli.main(["fetch", "--url", url]) == 0
    assert cli.main(["ingest", "--max-tokens", "4", "--embedder", "hash"]) == 0
    capsys.readouterr()
    try:
        code = cli.main(["search", "caf\udce9", "--mode", "full-text"])
    except UnicodeError as e:
        pytest.fail(f"traceback instead of a message: {e!r}")
    assert code in (0, 1)
