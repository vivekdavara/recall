from __future__ import annotations

import socket
from pathlib import Path

import pytest
from archives import COMMIT, FILES, PAGE, make_archive
from conftest import TEST_DATABASE_URL

from recall import cli
from recall.corpus import fetch, source
from recall.corpus.fetch import Manifest
from recall.embedding import HashEmbedder


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Manifest]:
    """A data directory and a manifest for a small local archive, which the CLI uses instead of the pinned ones."""
    archive = make_archive(tmp_path / "a.tar.gz")
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    monkeypatch.setattr(fetch, "load_manifest", lambda: manifest)
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    return archive.as_uri(), manifest


def test_fetch_verify_and_stats(setup: tuple[str, Manifest], capsys: pytest.CaptureFixture[str], tmp_path: Path):
    url, _ = setup
    assert cli.main(["fetch", "--url", url]) == 0
    assert "fetched 3 files" in capsys.readouterr().out
    assert cli.main(["fetch", "--url", url]) == 0
    assert "already up to date" in capsys.readouterr().out
    assert cli.main(["verify"]) == 0
    assert cli.main(["stats", "--max-tokens", "4"]) == 0
    out = capsys.readouterr().out
    assert "documents       1" in out and "chunks          3 (max_tokens 4)" in out
    assert "tokens a chunk  min 3 / p50 3 / p95 4 / max 4\nchunks at max   1\n" in out

    (tmp_path / "data/corpus/LICENSE").write_text("changed")
    assert cli.main(["verify"]) == 1
    captured = capsys.readouterr()
    assert "changed: LICENSE" in captured.out and "run recall fetch" in captured.err
    assert cli.main(["stats"]) == 1
    assert "doesn't match the manifest" in capsys.readouterr().err


def test_fetch_of_the_wrong_archive_fails_cleanly(
    setup: tuple[str, Manifest], capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    other = make_archive(tmp_path / "other.tar.gz", commit="d" * 40)
    assert cli.main(["fetch", "--url", other.as_uri()]) == 1
    assert "declares commit" in capsys.readouterr().err
    assert not (tmp_path / "data/corpus").exists()


def test_ingest_and_migrate(
    setup: tuple[str, Manifest], schema: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    url, _ = setup
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    assert cli.main(["migrate"]) == 0
    assert "applied 001_documents_and_chunks, 002_retrieval" in capsys.readouterr().out
    assert cli.main(["migrate"]) == 0
    assert "nothing to apply" in capsys.readouterr().out
    assert cli.main(["fetch", "--url", url]) == 0
    assert cli.main(["ingest", "--max-tokens", "4", "--embedder", "hash"]) == 0
    out = capsys.readouterr().out
    assert "1 added, 0 updated, 0 unchanged, 0 removed; 3 chunks written, 3 embedded (hash-v1-384)" in out
    monkeypatch.setenv("RECALL_EMBEDDER", "hash")
    assert cli.main(["ingest", "--max-tokens", "4"]) == 0
    assert "0 added, 0 updated, 1 unchanged, 0 removed; 0 chunks written, 0 embedded" in capsys.readouterr().out


def test_manifest_is_built_for_the_pinned_commit(
    setup: tuple[str, Manifest], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _, manifest = setup
    archive, out = str(tmp_path / "a.tar.gz"), tmp_path / "m.json"
    # The test archive isn't of the pinned commit, so it is refused...
    assert cli.main(["manifest", archive, "--out", str(out)]) == 1
    assert "declares commit" in capsys.readouterr().err and not out.exists()
    # ...until it is the pinned one.
    monkeypatch.setattr(source, "COMMIT", COMMIT)
    monkeypatch.setattr(source, "TAG", "t")
    assert cli.main(["manifest", archive, "--out", str(out)]) == 0
    assert Manifest.from_json(out.read_text()) == manifest
    assert "3 files, 1 pages" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-3", "x"])
def test_max_tokens_must_be_a_positive_number(value: str, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as e:
        cli.main(["stats", "--max-tokens", value])
    assert e.value.code == 2


def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert [cli.percentile(values, q) for q in (1, 50, 95, 100)] == [1, 50, 95, 100]
    assert cli.percentile([7], 50) == 7
    with pytest.raises(ValueError):
        cli.percentile([], 50)


def test_manifest_goes_to_standard_output_without_out(
    setup: tuple[str, Manifest], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _, manifest = setup
    monkeypatch.setattr(source, "COMMIT", COMMIT)
    monkeypatch.setattr(source, "TAG", "t")
    assert cli.main(["manifest", str(tmp_path / "a.tar.gz")]) == 0
    assert Manifest.from_json(capsys.readouterr().out) == manifest


def test_show_prints_a_pages_chunks(setup: tuple[str, Manifest], capsys: pytest.CaptureFixture[str]):
    url, _ = setup
    assert cli.main(["fetch", "--url", url]) == 0
    capsys.readouterr()
    # The page is "# A { #a }\n\n```python\nx = 1\n```\n": the fence starts at 12, and its opener is 4 tokens.
    assert cli.main(["show", "tutorial/a.md", "--max-tokens", "4"]) == 0
    assert capsys.readouterr().out == (
        "tutorial/a.md: A <https://fastapi.tiangolo.com/tutorial/a/>\n"
        "\n--- chunk 0: A #a; [12:21], 4 tokens\n```python"
        "\n\n--- chunk 1: A #a; [22:27], 3 tokens\nx = 1"
        "\n\n--- chunk 2: A #a; [28:31], 3 tokens\n```\n"
    )
    assert cli.main(["show", "index.md"]) == 1  # not in this small corpus
    capsys.readouterr()
    assert cli.main(["show", "nope.md"]) == 1
    assert "no page 'nope.md'" in capsys.readouterr().err


def test_a_database_that_isnt_there_is_one_line_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("RECALL_DATABASE_URL", f"postgresql://localhost:{port}/recall?connect_timeout=2")
    assert cli.main(["migrate"]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"recall: can't use the database localhost:{port}/recall: ") and err.count("\n") == 1


def test_a_page_that_cant_be_normalized_is_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # Building a manifest normalizes every page (to learn what it includes), so this is where a bad page shows.
    files = {**FILES, PAGE: b"# A\n\n{* ../../docs_src/a/t1.py ln[5:9] *}\n"}
    archive = make_archive(tmp_path / "bad.tar.gz", files=files)
    monkeypatch.setattr(source, "COMMIT", COMMIT)
    assert cli.main(["manifest", str(archive)]) == 1
    assert capsys.readouterr().err == f"recall: {PAGE}: line range 5:9 outside 1..1\n"


def test_show_names_the_text_before_the_first_heading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    archive = make_archive(tmp_path / "p.tar.gz", files={**FILES, PAGE: b"Before any heading.\n"})
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    monkeypatch.setattr(fetch, "load_manifest", lambda: manifest)
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    assert cli.main(["fetch", "--url", archive.as_uri()]) == 0
    capsys.readouterr()
    assert cli.main(["show", "tutorial/a.md"]) == 0
    assert "--- chunk 0: (before the first heading); [0:19], 4 tokens\nBefore any heading.\n" in capsys.readouterr().out


def test_search(
    setup: tuple[str, Manifest], schema: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    url, _ = setup
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    assert cli.main(["fetch", "--url", url]) == 0
    assert cli.main(["search", "x", "--mode", "full-text"]) == 1
    assert capsys.readouterr().err.endswith("/recall_test has no corpus or an older schema: run recall ingest\n")
    assert cli.main(["migrate"]) == 0
    assert cli.main(["search", "x", "--embedder", "hash"]) == 1
    assert "no embedded corpus: run recall ingest" in capsys.readouterr().err
    assert cli.main(["ingest", "--max-tokens", "4", "--embedder", "hash"]) == 0
    capsys.readouterr()

    assert cli.main(["search", "x = 1", "--embedder", "hash", "-k", "2", "--width", "5"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "1. tutorial/a.md#a  A"
    assert out[1].startswith("   score 0.0") and "(full-text #1, vector #" in out[1]
    assert out[2] in {"   x = 1", "   ```py..."} and len(out) == 6

    # Full-text search loads no model: an unusable one doesn't matter.
    monkeypatch.setattr(cli, "make_embedder", lambda kind: pytest.fail("loaded a model"))
    assert cli.main(["search", "x", "--mode", "full-text"]) == 0
    assert "(full-text #1)" in capsys.readouterr().out
    assert cli.main(["search", "zebra", "--mode", "full-text"]) == 0
    assert capsys.readouterr().out == "no results\n"


def test_search_with_vectors_of_another_model_fails_cleanly(
    setup: tuple[str, Manifest], schema: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    url, _ = setup
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    assert cli.main(["fetch", "--url", url]) == 0
    assert cli.main(["ingest", "--embedder", "hash"]) == 0
    capsys.readouterr()

    class Other(HashEmbedder):
        @property
        def name(self) -> str:
            return "other"

    monkeypatch.setattr(cli, "make_embedder", lambda kind: Other())
    assert cli.main(["search", "x"]) == 1
    err = capsys.readouterr().err
    assert (
        err
        == "recall: the stored vectors are from hash-v1-384, not other: search with that embedder, or ingest again\n"
    )


def test_the_model_embedder_without_sentence_transformers_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import builtins

    real_import = builtins.__import__

    def no_model(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_model)
    assert cli.main(["search", "x", "--embedder", "model"]) == 1
    assert "needs sentence-transformers: pip install -e '.[embed]'" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["-k", "--candidates"])
def test_search_refuses_lists_longer_than_pgvector_can_search(option: str, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as e:
        cli.main(["search", "x", option, "1001"])
    assert e.value.code == 2
    assert "must be at most 1000, not 1001" in capsys.readouterr().err


def test_an_invalid_recall_embedder_stops_the_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("RECALL_EMBEDDER", "Hash")
    with pytest.raises(SystemExit) as e:
        cli.main(["ingest"])
    assert e.value.code == 2
    assert "RECALL_EMBEDDER must be one of model, hash, not 'Hash'" in capsys.readouterr().err


def test_search_accepts_lists_of_1000():
    args = cli.parser().parse_args(["search", "x", "-k", "1000", "--candidates", "1000"])
    assert (args.k, args.candidates) == (1000, 1000)
