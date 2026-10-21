from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest
from archives import COMMIT, FILES, PAGE, make_archive

from recall.corpus import fetch, source
from recall.corpus.fetch import CorpusError, Manifest, ManifestFile


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return make_archive(tmp_path / "a.tar.gz")


@pytest.fixture
def manifest(archive: Path) -> Manifest:
    return fetch.build_manifest(archive, commit=COMMIT, tag="t")


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


def test_the_manifest_lists_kept_pages_their_includes_and_the_license(manifest: Manifest):
    assert [f.path for f in manifest.files] == ["LICENSE", PAGE, "docs_src/a/t1.py"]
    assert manifest.pages() == ["tutorial/a.md"]
    assert manifest.files[2] == ManifestFile("docs_src/a/t1.py", 6, hashlib.sha256(b"x = 1\n").hexdigest())


def test_a_manifest_round_trips_through_json(manifest: Manifest):
    assert Manifest.from_json(manifest.to_json()) == manifest


def test_extract_writes_exactly_the_listed_files(archive: Path, manifest: Manifest, tmp_path: Path):
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
        "LICENSE",
        PAGE,
        "docs_src/a/t1.py",
    ]
    assert (dest / "docs_src/a/t1.py").read_bytes() == b"x = 1\n"
    assert fetch.verify(dest, manifest) == []
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("commit", [None, "d" * 40])
def test_an_archive_of_another_commit_is_refused(tmp_path: Path, manifest: Manifest, commit: str | None):
    other = make_archive(tmp_path / "other.tar.gz", commit=commit)
    with pytest.raises(CorpusError, match="declares commit"):
        fetch.extract(other, manifest, tmp_path / "corpus")
    assert not (tmp_path / "corpus").exists()


def test_a_changed_file_fails_and_leaves_the_old_corpus_in_place(archive: Path, manifest: Manifest, tmp_path: Path):
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    tampered = make_archive(tmp_path / "t.tar.gz", files={**FILES, "docs_src/a/t1.py": b"x = 2\n"})
    with pytest.raises(CorpusError, match=r"docs_src/a/t1.py differs from the manifest"):
        fetch.extract(tampered, manifest, dest)
    assert fetch.verify(dest, manifest) == []
    assert leftovers(tmp_path) == []


def test_a_missing_file_fails(tmp_path: Path, manifest: Manifest):
    files = {k: v for k, v in FILES.items() if k != "LICENSE"}
    with pytest.raises(CorpusError, match="the archive has no LICENSE"):
        fetch.extract(make_archive(tmp_path / "m.tar.gz", files=files), manifest, tmp_path / "corpus")


def test_a_link_where_a_file_should_be_fails(tmp_path: Path, manifest: Manifest):
    files = {k: v for k, v in FILES.items() if k != "LICENSE"}
    link = tarfile.TarInfo("fastapi-x/LICENSE")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    with pytest.raises(CorpusError, match="not a regular file"):
        fetch.extract(make_archive(tmp_path / "l.tar.gz", files=files, extra=[link]), manifest, tmp_path / "c")


def test_two_members_for_one_path_fail(tmp_path: Path, manifest: Manifest):
    dup = tarfile.TarInfo("fastapi-x/LICENSE")
    dup.size = 4
    with pytest.raises(CorpusError, match="two members"):
        fetch.extract(make_archive(tmp_path / "d.tar.gz", extra=[dup]), manifest, tmp_path / "c")


def test_a_member_outside_the_top_directory_fails(tmp_path: Path, manifest: Manifest):
    stray = tarfile.TarInfo("elsewhere/x")
    stray.size = 1
    with pytest.raises(CorpusError, match="outside the top directory"):
        fetch.extract(make_archive(tmp_path / "s.tar.gz", extra=[stray]), manifest, tmp_path / "c")


@pytest.mark.parametrize("path", ["", "/etc/passwd", "../x", "a/../b", "a//b", "./a", "a/", "a\\b"])
def test_manifest_paths_must_be_plain_relative_paths(path: str):
    with pytest.raises(CorpusError, match="not allowed"):
        Manifest("r", "t", COMMIT, (ManifestFile(path, 1, "0" * 64),)).check()


def test_manifest_paths_must_be_sorted_and_unique():
    f = ManifestFile("a", 1, "0" * 64)
    g = ManifestFile("b", 1, "0" * 64)
    for files in [(g, f), (f, f)]:
        with pytest.raises(CorpusError, match="unique and sorted"):
            Manifest("r", "t", COMMIT, files).check()


def test_verify_names_missing_changed_and_extra_files(archive: Path, manifest: Manifest, tmp_path: Path):
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    (dest / "LICENSE").unlink()
    (dest / "docs_src/a/t1.py").write_bytes(b"x = 3\n")
    (dest / "docs_src/a/new.py").write_bytes(b"")
    assert fetch.verify(dest, manifest) == [
        "missing: LICENSE",
        "changed: docs_src/a/t1.py",
        "not in the manifest: docs_src/a/new.py",
    ]


def test_fetch_downloads_once_and_repairs_a_damaged_corpus(archive: Path, manifest: Manifest, tmp_path: Path):
    dest = tmp_path / "data" / "corpus"
    url = archive.as_uri()
    assert fetch.fetch(dest, manifest, url) is True
    assert fetch.fetch(dest, manifest, url) is False
    (dest / "LICENSE").write_bytes(b"changed\n")
    assert fetch.fetch(dest, manifest, url) is True
    assert fetch.verify(dest, manifest) == []
    assert leftovers(dest.parent) == []


def test_download_refuses_an_archive_over_the_limit(archive: Path, tmp_path: Path):
    with pytest.raises(CorpusError, match="larger than"):
        fetch.download(archive.as_uri(), tmp_path / "x", max_bytes=100)


def test_a_page_whose_include_is_missing_has_no_manifest(tmp_path: Path):
    files = {k: v for k, v in FILES.items() if k != "docs_src/a/t1.py"}
    with pytest.raises(CorpusError, match=r"no docs_src/a/t1\.py"):
        fetch.build_manifest(make_archive(tmp_path / "i.tar.gz", files=files), commit=COMMIT)


def test_the_committed_manifest_is_for_the_pinned_commit():
    manifest = fetch.load_manifest()
    assert (manifest.repository, manifest.tag, manifest.commit) == (source.REPOSITORY, source.TAG, source.COMMIT)
    assert len(manifest.pages()) == 125
    assert "LICENSE" in [f.path for f in manifest.files]
    assert not [p for p in manifest.pages() if p.startswith("reference/") or p in source.EXCLUDED_PAGES]


@pytest.mark.parametrize(
    ("page", "url"),
    [
        ("index.md", "https://fastapi.tiangolo.com/"),
        ("tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/"),
        ("tutorial/query-params.md", "https://fastapi.tiangolo.com/tutorial/query-params/"),
        ("async.md", "https://fastapi.tiangolo.com/async/"),
    ],
)
def test_page_urls_follow_the_sites_directory_urls(page: str, url: str):
    assert source.page_url(page) == url


def test_which_paths_are_kept_pages():
    assert source.page_of("docs/en/docs/tutorial/a.md") == "tutorial/a.md"
    for path in ["docs/en/docs/release-notes.md", "docs/en/docs/reference/x.md", "docs/es/docs/a.md", "docs/en/a.md"]:
        assert source.page_of(path) is None, path
    assert source.page_of("docs/en/docs/img/a.png") is None


def test_verify_of_a_corpus_never_fetched_lists_every_file_as_missing(manifest: Manifest, tmp_path: Path):
    assert fetch.verify(tmp_path / "nothing", manifest) == [f"missing: {f.path}" for f in manifest.files]


@pytest.mark.parametrize("directory_first", [True, False])
def test_a_directory_and_a_file_at_one_path_fail(tmp_path: Path, manifest: Manifest, directory_first: bool):
    d = tarfile.TarInfo("fastapi-x/LICENSE")
    d.type = tarfile.DIRTYPE
    files = {k: v for k, v in FILES.items() if k != "LICENSE"}
    f = tarfile.TarInfo("fastapi-x/LICENSE")
    f.size = 4
    extra = [d, f] if directory_first else [f, d]
    with pytest.raises(CorpusError, match="two members"):
        fetch.extract(make_archive(tmp_path / "df.tar.gz", files=files, extra=extra), manifest, tmp_path / "c")


def test_verify_reports_a_listed_file_replaced_by_a_link_to_the_same_bytes(
    archive: Path, manifest: Manifest, tmp_path: Path
):
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    copy = tmp_path / "copy"
    copy.write_bytes((dest / "LICENSE").read_bytes())
    (dest / "LICENSE").unlink()
    (dest / "LICENSE").symlink_to(copy)
    assert fetch.verify(dest, manifest) == ["missing: LICENSE"]


def test_a_file_that_isnt_an_archive_is_a_corpus_error(tmp_path: Path, manifest: Manifest):
    (tmp_path / "x.tar.gz").write_text("<html>rate limited</html>")
    with pytest.raises(CorpusError, match="not a tar archive"):
        fetch.extract(tmp_path / "x.tar.gz", manifest, tmp_path / "c")


def test_a_member_named_to_escape_is_never_written(archive: Path, manifest: Manifest, tmp_path: Path):
    # Members are read by the manifest's paths, so a member whose name climbs out is simply never used.
    evil = tarfile.TarInfo("fastapi-x/../../escaped.txt")
    evil.size = 4
    dest = tmp_path / "deep" / "corpus"
    fetch.extract(make_archive(tmp_path / "e.tar.gz", extra=[evil]), manifest, dest)
    assert fetch.verify(dest, manifest) == []
    assert not list(tmp_path.rglob("escaped.txt"))


# The test's own list: parametrizing over source.EXCLUDED_PAGES would drop a case whenever an exclusion is deleted.
EXCLUDED = [
    "_llm-test.md",
    "external-links.md",
    "fastapi-people.md",
    "management.md",
    "newsletter.md",
    "release-notes.md",
    "translation-banner.md",
]


def test_the_exclusions_are_the_designed_ones():
    assert sorted(source.EXCLUDED_PAGES) == EXCLUDED
    assert list(source.EXCLUDED_DIRS) == ["reference/"]


@pytest.mark.parametrize("page", EXCLUDED)
def test_every_excluded_page_is_not_kept(page: str):
    assert source.page_of("docs/en/docs/" + page) is None
    assert source.page_of("docs/en/docs/tutorial/" + page) == "tutorial/" + page  # only at the top


def test_a_link_where_the_corpus_should_be_is_replaced(archive: Path, manifest: Manifest, tmp_path: Path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("not ours")
    dest = tmp_path / "corpus"
    dest.symlink_to(elsewhere)
    assert fetch.verify(dest, manifest)[-1] == f"a link, not written by fetch: {dest}"
    fetch.extract(archive, manifest, dest)
    assert not dest.is_symlink() and fetch.verify(dest, manifest) == []
    assert (elsewhere / "keep.txt").read_text() == "not ours"  # the link went, not what it pointed at
    assert leftovers(tmp_path) == []


def test_verify_reports_an_extra_link(archive: Path, manifest: Manifest, tmp_path: Path):
    dest = tmp_path / "corpus"
    fetch.extract(archive, manifest, dest)
    (dest / "docs_src/a/alias.py").symlink_to(dest / "docs_src/a/t1.py")
    assert fetch.verify(dest, manifest) == ["a link, not written by fetch: docs_src/a/alias.py"]


def test_an_archive_that_isnt_there_is_a_corpus_error(tmp_path: Path, manifest: Manifest):
    with pytest.raises(CorpusError, match="not a tar archive"):
        fetch.extract(tmp_path / "missing.tar.gz", manifest, tmp_path / "c")


def test_a_download_that_fails_is_a_corpus_error(tmp_path: Path):
    with pytest.raises(CorpusError, match="can't download"):
        fetch.download((tmp_path / "missing.tar.gz").as_uri(), tmp_path / "x")


def test_a_dangling_link_where_the_corpus_should_be_is_replaced(archive: Path, manifest: Manifest, tmp_path: Path):
    dest = tmp_path / "corpus"
    dest.symlink_to(tmp_path / "gone")
    fetch.extract(archive, manifest, dest)
    assert not dest.is_symlink() and fetch.verify(dest, manifest) == []
    assert leftovers(tmp_path) == []
