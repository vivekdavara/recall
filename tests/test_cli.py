from __future__ import annotations

from pathlib import Path

import pytest
from archives import COMMIT, make_archive
from conftest import TEST_DATABASE_URL

from recall import cli
from recall.corpus import fetch, source
from recall.corpus.fetch import Manifest


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
    assert "applied 001_documents_and_chunks" in capsys.readouterr().out
    assert cli.main(["migrate"]) == 0
    assert "nothing to apply" in capsys.readouterr().out
    assert cli.main(["fetch", "--url", url]) == 0
    assert cli.main(["ingest", "--max-tokens", "4"]) == 0
    assert "1 added, 0 updated, 0 unchanged, 0 removed; 3 chunks written" in capsys.readouterr().out
    assert cli.main(["ingest", "--max-tokens", "4"]) == 0
    assert "0 added, 0 updated, 1 unchanged" in capsys.readouterr().out


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
