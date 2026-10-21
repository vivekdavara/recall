"""Checks on the real, pinned corpus. They need `recall fetch` first: CI's corpus job fetches and runs them with
RECALL_REQUIRE_CORPUS=1, so there they fail rather than skip when the corpus is missing."""

from __future__ import annotations

import hashlib
import os
import re
from itertools import pairwise
from pathlib import Path

import pytest

from recall import chunking, markdown
from recall.corpus import fetch, normalize, source
from recall.corpus.load import Document, load

CORPUS = Path(os.environ.get("RECALL_DATA_DIR", "data")) / "corpus"
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    if not CORPUS.is_dir():
        if os.environ.get("RECALL_REQUIRE_CORPUS") == "1":
            pytest.fail(f"no corpus at {CORPUS}: run recall fetch")
        pytest.skip(f"no corpus at {CORPUS} (recall fetch); CI's corpus job runs these")
    return load(CORPUS, fetch.load_manifest())


def outside_fences(text: str) -> list[str]:
    return [text[ln.start : ln.end] for ln in markdown.scan(text) if ln.fence is None]


def test_every_kept_page_loads_with_a_title_of_its_own(documents: list[Document]):
    assert len(documents) == 125
    assert not [d.path for d in documents if d.title == d.path]


def test_normalizing_a_normalized_page_changes_nothing(documents: list[Document]):
    manifest = fetch.load_manifest()
    listed = {f.path for f in manifest.files}

    def read(path: str) -> str:
        assert path in listed
        return CORPUS.joinpath(path).read_text(encoding="utf-8")

    for d in documents:
        assert normalize.normalize(d.path, d.content, read) == d.content, d.path


def test_no_documentation_build_syntax_is_left_outside_code(documents: list[Document]):
    for d in documents:
        assert not d.content.startswith("---\n"), d.path  # front matter
        for line in outside_fences(d.content):
            assert not line.lstrip().startswith(("{*", "{!", "///")), (d.path, line)
            assert "<!--" not in line, (d.path, line)
            assert not _TAG.search(_CODE_SPAN.sub("", line)), (d.path, line)


def test_the_manifest_holds_exactly_what_the_pages_use(documents: list[Document]):
    manifest = fetch.load_manifest()

    def read(path: str) -> str:
        return CORPUS.joinpath(path).read_text(encoding="utf-8")

    used = {source.LICENSE_PATH}
    for page in manifest.pages():
        repo_path = source.repo_path_of(page)
        used.add(repo_path)
        used.update(normalize.includes(page, read(repo_path), read))
    assert used == {f.path for f in manifest.files}


@pytest.mark.parametrize("max_tokens", [256, 32])
def test_every_page_chunks_within_the_invariants(documents: list[Document], max_tokens: int):
    for d in documents:
        text = d.content
        chunks = chunking.chunk_document(text, max_tokens)
        assert chunks, d.path
        sections = chunking.sections(text)
        heading_lines = [(a.end, b.start) for a, b in pairwise(sections)]
        covered = bytearray(len(text))
        for c in chunks:
            assert c.text == text[c.start : c.end]  # C1
            assert 1 <= c.token_count == chunking.count_tokens(c.text) <= max_tokens  # C4
            covered[c.start : c.end] = b"\x01" * (c.end - c.start)
        for a, b in pairwise(chunks):
            assert a.end <= b.start  # C2
            if not any(a.end <= s.start and s.start <= b.start for s in sections):  # one section: C8
                assert a.token_count + b.token_count > max_tokens, (d.path, a.ordinal)
        for i, ch in enumerate(text):  # C3
            if not covered[i] and not ch.isspace():
                assert any(s <= i < e for s, e in heading_lines), (d.path, i)
        fences: dict[int, list[markdown.Line]] = {}
        for ln in markdown.scan(text):
            if ln.fence is not None:
                fences.setdefault(ln.fence, []).append(ln)
        for lines in fences.values():  # C6
            start, end = lines[0].start, lines[-1].end
            start += len(text[start:end]) - len(text[start:end].lstrip())
            if chunking.count_tokens(text[start:end]) <= max_tokens:
                assert any(c.start <= start and end <= c.end for c in chunks), (d.path, text[start:end][:60])


def test_the_chunks_are_the_ones_this_chunker_version_makes(documents: list[Document]):
    # A pin on the whole corpus. If a change to the chunker changes these numbers, raise CHUNKER_VERSION (so ingest
    # re-chunks stored corpora) and re-pin them from the new output. A change to normalization changes the
    # documents' content hashes, so ingest re-chunks those documents by itself: re-pin only.
    chunks = [c for d in documents for c in chunking.chunk_document(d.content)]
    assert chunking.CHUNKER_VERSION == 2
    assert (len(chunks), sum(c.token_count for c in chunks)) == (1503, 217390)
    # Every chunk's span, so a change that moves a cut without changing the totals shows too.
    spans = hashlib.sha256()
    for d in documents:
        for c in chunking.chunk_document(d.content):
            spans.update(f"{d.path}:{c.start}:{c.end}\n".encode())
    assert spans.hexdigest() == "50b5796ff78bb1d486f25399d087ad2baee6af2f94d14a3c3feaf73fe3e4a917"
