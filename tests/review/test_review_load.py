"""Review findings for recall.corpus.load and fetch.verify. Each test fails on 40a7144."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from archives import COMMIT, FILES, PAGE, make_archive

from recall.chunking import chunk_document
from recall.corpus import fetch
from recall.corpus.load import load


def test_a_page_saved_with_a_utf8_bom_keeps_its_heading_and_title(tmp_path: Path):
    """A UTF-8 byte order mark is not page text (mkdocs reads pages as utf-8-sig). Today it is kept, so the
    first line "\\ufeff# A { #a }" is not a heading: the page gets its path as title and the H1 becomes chunk
    text with an empty heading path."""
    archive = make_archive(tmp_path / "a.tar.gz", files={**FILES, PAGE: b"\xef\xbb\xbf# A { #a }\n\ntext\n"})
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    fetch.extract(archive, manifest, tmp_path / "corpus")
    [doc] = load(tmp_path / "corpus", manifest)
    assert doc.title == "A"
    assert [(c.heading_path, c.anchor, c.text) for c in chunk_document(doc.content)] == [(("A",), "a", "text")]


def test_verify_reports_an_extra_file_behind_a_symlinked_directory(tmp_path: Path):
    """`recall verify` reports extra files (DESIGN.md "Integrity"). A directory replaced by a symlink is
    skipped by the extra-file walk (`p.is_dir()` follows the link, rglob doesn't descend), so neither the link
    nor the extra file behind it is reported and `fetch` says the corpus is up to date. Low severity: only a
    hand-made symlink gets there."""
    archive = make_archive(tmp_path / "a.tar.gz")
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    shutil.move(dest / "docs_src", tmp_path / "elsewhere")
    os.symlink(tmp_path / "elsewhere", dest / "docs_src")
    (tmp_path / "elsewhere/a/extra.py").write_text("extra\n")
    assert fetch.verify(dest, manifest) != []
