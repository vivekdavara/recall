"""Read an extracted corpus into normalized documents, after checking it against the manifest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from recall.corpus import normalize, source
from recall.corpus.fetch import CorpusError, Manifest, verify


@dataclass(frozen=True)
class Document:
    path: str  # the page path, e.g. tutorial/query-params.md
    title: str
    url: str
    content: str  # normalized Markdown: every stored offset is into this text

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def load(corpus_dir: Path, manifest: Manifest) -> list[Document]:
    """Every kept page, normalized, in path order. Fails if the corpus doesn't match the manifest."""
    problems = verify(corpus_dir, manifest)
    if problems:
        more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
        raise CorpusError(f"{corpus_dir} doesn't match the manifest: {'; '.join(problems[:5])}{more}; run recall fetch")
    listed = {f.path for f in manifest.files}

    def read(path: str) -> str:
        # Only what the manifest lists (and so was checked) may be read, includes included.
        if path not in listed:
            raise FileNotFoundError(path)
        try:
            return corpus_dir.joinpath(*PurePosixPath(path).parts).read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise CorpusError(f"{path} is not UTF-8: {e}") from e

    documents = []
    for page in manifest.pages():
        content = normalize.normalize(page, read(source.repo_path_of(page)), read)
        documents.append(Document(page, normalize.title(content, page), source.page_url(page), content))
    return documents
