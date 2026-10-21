"""Review findings for the recall command's error handling. Fails on 40a7144."""

from __future__ import annotations

from pathlib import Path

import pytest
from archives import COMMIT, make_archive

from recall import cli
from recall.corpus import fetch


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    archive = make_archive(tmp_path / "a.tar.gz")
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    monkeypatch.setattr(fetch, "load_manifest", lambda: manifest)
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.mark.parametrize("kind", ["not an archive", "no such file"])
def test_fetch_of_a_bad_download_fails_cleanly(data: Path, capsys: pytest.CaptureFixture[str], kind: str):
    """Like the wrong-commit case (test_cli.test_fetch_of_the_wrong_archive_fails_cleanly), a URL that isn't a
    tar archive, or can't be fetched, should give exit 1 and a `recall: ...` line, not a traceback. Today
    tarfile.ReadError and urllib's URLError escape main() (psycopg.OperationalError does too, for migrate and
    ingest with the database down)."""
    target = data / "page.html"
    if kind == "not an archive":
        target.write_text("<html>rate limited</html>")
    assert cli.main(["fetch", "--url", target.as_uri()]) == 1
    assert capsys.readouterr().err.startswith("recall: ")
    assert not (data / "data/corpus").exists()
