"""Review findings for recall.corpus.fetch (DESIGN.md "The corpus", "Integrity"). Each test fails on 40a7144."""

from __future__ import annotations

from pathlib import Path

import pytest
from archives import COMMIT, FILES, PAGE, make_archive

from recall.corpus import fetch
from recall.corpus.load import load


def built_and_loaded(tmp_path: Path, page: bytes) -> list[str]:
    """Build a manifest from an archive with this page, extract it, and load it; the documents' contents."""
    archive = make_archive(tmp_path / "a.tar.gz", files={**FILES, PAGE: page})
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    fetch.extract(archive, manifest, tmp_path / "corpus")
    return [d.content for d in load(tmp_path / "corpus", manifest)]


def test_a_manifest_built_from_a_crlf_page_lists_its_includes(tmp_path: Path):
    """`recall manifest` must list every file a page includes, whatever its line endings.

    `normalize` turns CRLF into LF before matching includes, but `build_manifest` calls `normalize.includes` on
    the raw bytes, where `[ \\t]*$` doesn't match before the "\\r". The manifest omits docs_src/a/t1.py and the
    corpus it describes can't be loaded ("included file not found").
    """
    assert built_and_loaded(tmp_path, b"# A\r\n\r\n{* ../../docs_src/a/t1.py *}\r\n") == [
        "# A\n\n```python\nx = 1\n```\n"
    ]


@pytest.mark.parametrize(
    "line",
    [b"{* ../../docs_src/a/t1.py *} <!-- the first example -->", b"<!-- example --> {* ../../docs_src/a/t1.py *}"],
)
def test_a_manifest_lists_an_include_that_shares_its_line_with_a_comment(tmp_path: Path, line: bytes):
    """`normalize` strips comments before matching a code include, `includes` doesn't: the two must agree on
    which files a page uses, or the manifest misses one and `load` fails on a corpus that `verify` accepts."""
    contents = built_and_loaded(tmp_path, b"# A\n\n" + line + b"\n")
    assert "x = 1" in contents[0]


def test_a_failed_rename_into_place_keeps_the_old_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """All or nothing: if putting the new corpus in place fails, data/corpus must still hold the old one.

    `_replace` renames the old corpus aside and then renames staging into place; if the second rename fails the
    old corpus is left at `.corpus.old-<pid>` and data/corpus is gone (and nothing ever cleans the leftover).
    """
    archive = make_archive(tmp_path / "a.tar.gz")
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)

    real_rename = Path.rename
    calls: list[Path] = []

    def rename(self: Path, target: Path) -> Path:
        calls.append(self)
        if len(calls) == 2:  # staging -> dest
            raise OSError("simulated failure")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError, match="simulated"):
        fetch.extract(archive, manifest, dest)
    monkeypatch.undo()
    assert fetch.verify(dest, manifest) == []
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_extract_over_a_file_where_the_corpus_should_be_succeeds_without_leftovers(tmp_path: Path):
    """If data/corpus is a stray file, extract replaces it; today the corpus is placed, then `shutil.rmtree` of
    the renamed file raises NotADirectoryError, so the command fails after succeeding and leaves `.corpus.old-*`."""
    archive = make_archive(tmp_path / "a.tar.gz")
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    dest = tmp_path / "corpus"
    dest.write_text("not a directory")
    fetch.extract(archive, manifest, dest)
    assert fetch.verify(dest, manifest) == []
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []
