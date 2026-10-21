from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from archives import COMMIT, FILES, make_archive

from recall.corpus import fetch
from recall.corpus.fetch import CorpusError, Manifest, ManifestFile
from recall.corpus.load import load
from recall.corpus.normalize import NormalizeError

MORE = {
    **FILES,
    "docs/en/docs/index.md": b"# FastAPI { #fastapi }\n\n/// tip\n\nFast.\n\n///\n",
    "docs/en/docs/tutorial/no-title.md": b"Just text, <b>bold</b>.\n",
}


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Manifest]:
    archive = make_archive(tmp_path / "a.tar.gz", files=MORE)
    manifest = fetch.build_manifest(archive, commit=COMMIT, tag="t")
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    return dest, manifest


def test_load_gives_each_kept_page_normalized_with_its_title_and_url(corpus: tuple[Path, Manifest]):
    docs = load(*corpus)
    assert [(d.path, d.title, d.url) for d in docs] == [
        ("index.md", "FastAPI", "https://fastapi.tiangolo.com/"),
        ("tutorial/a.md", "A", "https://fastapi.tiangolo.com/tutorial/a/"),
        ("tutorial/no-title.md", "tutorial/no-title.md", "https://fastapi.tiangolo.com/tutorial/no-title/"),
    ]
    assert docs[0].content == "# FastAPI { #fastapi }\n\nTip:\n\nFast.\n\n\n"
    assert docs[1].content == "# A { #a }\n\n```python\nx = 1\n```\n"
    assert docs[2].content == "Just text, bold.\n"


def test_a_corpus_that_doesnt_match_the_manifest_is_refused(corpus: tuple[Path, Manifest]):
    dest, manifest = corpus
    (dest / "docs_src/a/t1.py").write_text("x = 2\n")
    with pytest.raises(CorpusError, match=r"changed: docs_src/a/t1.py; run recall fetch"):
        load(dest, manifest)


def test_only_files_in_the_manifest_can_be_included(corpus: tuple[Path, Manifest]):
    dest, manifest = corpus
    # A manifest without the include (and a corpus without the file): the page can't be built from it.
    smaller = Manifest(
        manifest.repository, manifest.tag, manifest.commit, tuple(f for f in manifest.files if "docs_src" not in f.path)
    )
    (dest / "docs_src/a/t1.py").unlink()
    with pytest.raises(NormalizeError, match=r"included file not found: docs_src/a/t1\.py"):
        load(dest, smaller)


def test_a_page_that_isnt_utf8_is_an_error(tmp_path: Path):
    files = {**FILES, "docs/en/docs/bad.md": b"caf\xe9\n"}
    archive = make_archive(tmp_path / "b.tar.gz", files=files)
    with pytest.raises(CorpusError, match=r"docs/en/docs/bad\.md is not UTF-8"):
        fetch.build_manifest(archive, commit=COMMIT)


def test_an_included_file_that_isnt_utf8_has_no_manifest(tmp_path: Path):
    files = {**FILES, "docs_src/a/t1.py": b"caf\xe9 = 1\n"}
    with pytest.raises(CorpusError, match=r"docs_src/a/t1\.py is not UTF-8"):
        fetch.build_manifest(make_archive(tmp_path / "c.tar.gz", files=files), commit=COMMIT)


def test_load_refuses_a_listed_file_that_isnt_utf8(tmp_path: Path):
    # Only a hand-made manifest can list one (build_manifest refuses it): load still says what is wrong.
    files = {**FILES, "docs_src/a/t1.py": b"caf\xe9 = 1\n"}
    good = fetch.build_manifest(make_archive(tmp_path / "g.tar.gz"), commit=COMMIT)
    bad = Manifest(
        good.repository,
        good.tag,
        good.commit,
        tuple(
            ManifestFile(f.path, len(files[f.path]), hashlib.sha256(files[f.path]).hexdigest()) if "t1" in f.path else f
            for f in good.files
        ),
    )
    archive = make_archive(tmp_path / "c.tar.gz", files=files)
    fetch.extract(archive, bad, tmp_path / "corpus")
    with pytest.raises(CorpusError, match=r"docs_src/a/t1\.py is not UTF-8"):
        load(tmp_path / "corpus", bad)
