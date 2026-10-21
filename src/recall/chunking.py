"""Split a normalized Markdown document into chunks: sections at headings, blocks at blank lines, packed by size.

DESIGN.md "Chunking" is the specification; the invariants C1 to C7 named below are defined there.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterator
from dataclasses import dataclass

from recall import markdown

DEFAULT_MAX_TOKENS = 256

# Raise this whenever a change here can change any chunk: ingest then re-chunks every stored document.
CHUNKER_VERSION = 1

# A token is a run of word characters or one other non-whitespace character. Every non-whitespace character is in
# exactly one token and no token contains whitespace, so cutting text at whitespace or between two tokens splits
# its count exactly: count(a + b) == count(a) + count(b). The packing below relies on that to keep C4 exact.
_TOKEN = re.compile(r"\w+|[^\w\s]")
_RUN = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN.finditer(text))


@dataclass(frozen=True)
class Section:
    """A heading's section: its body runs from the line after the heading to the next heading (or the end).

    The text before the first heading is a section with level 0, an empty heading path and no anchor.
    """

    level: int
    heading_path: tuple[str, ...]
    anchor: str | None
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    heading_path: tuple[str, ...]
    anchor: str | None
    start: int
    end: int
    text: str
    token_count: int


def sections(text: str) -> list[Section]:
    """The document's sections in order. Sections with an empty body are included."""
    found: list[Section] = []
    stack: list[tuple[int, str]] = []
    level, anchor, body_start = 0, None, 0
    path_now: tuple[str, ...] = ()
    for line in markdown.scan(text):
        if line.fence is not None:
            continue
        h = markdown.heading(text[line.start : line.end])
        if h is None:
            continue
        found.append(Section(level, path_now, anchor, body_start, line.start))
        while stack and stack[-1][0] >= h.level:
            stack.pop()
        stack.append((h.level, h.title))
        level, path_now, anchor = h.level, tuple(t for _, t in stack), h.anchor
        body_start = min(line.end + 1, len(text))
    found.append(Section(level, path_now, anchor, body_start, len(text)))
    return found


def chunk_document(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Chunks covering every non-whitespace character of text outside heading lines (C3), in order (C2)."""
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be at least 1, not {max_tokens}")
    lines = markdown.scan(text)
    starts = [ln.start for ln in lines]
    chunks: list[Chunk] = []
    for section in sections(text):
        first = bisect.bisect_left(starts, section.start)
        last = bisect.bisect_left(starts, section.end, lo=first)
        body = [ln for ln in lines[first:last] if ln.end <= section.end]
        for start, end, n in _pack(_pieces(text, body, max_tokens), max_tokens):
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    heading_path=section.heading_path,
                    anchor=section.anchor,
                    start=start,
                    end=end,
                    text=text[start:end],
                    token_count=n,
                )
            )
    return chunks


def _blocks(text: str, lines: list[markdown.Line]) -> Iterator[list[markdown.Line]]:
    """Group a section's lines into blocks: blank lines separate blocks, and a fenced code block is its own block."""
    block: list[markdown.Line] = []
    for line in lines:
        if line.fence is None and markdown.is_blank(text[line.start : line.end]):
            if block:
                yield block
            block = []
            continue
        if block and block[-1].fence != line.fence:
            yield block
            block = []
        block.append(line)
    if block:
        yield block


def _pieces(text: str, lines: list[markdown.Line], max_tokens: int) -> Iterator[tuple[int, int, int]]:
    """Spans (start, end, tokens) that each fit in max_tokens and together hold every token of the lines.

    A block that fits is one piece; one that doesn't is split at line ends, a line at spaces, and a run of
    non-whitespace with more than max_tokens tokens between two tokens. Each piece starts and ends with a
    non-whitespace character.
    """
    for block in _blocks(text, lines):
        start, end = block[0].start, block[-1].end
        n = count_tokens(text[start:end])
        if n == 0:
            continue
        if n <= max_tokens:
            yield _trim(text, start, end, n)
            continue
        for line in block:
            yield from _line_pieces(text, line.start, line.end, max_tokens)


def _line_pieces(text: str, start: int, end: int, max_tokens: int) -> Iterator[tuple[int, int, int]]:
    n = count_tokens(text[start:end])
    if n == 0:
        return
    if n <= max_tokens:
        yield _trim(text, start, end, n)
        return
    for run in _RUN.finditer(text, start, end):
        r = count_tokens(run.group())
        if r <= max_tokens:
            yield run.start(), run.end(), r
        else:
            for token in _TOKEN.finditer(text, run.start(), run.end()):
                yield token.start(), token.end(), 1


def _trim(text: str, start: int, end: int, n: int) -> tuple[int, int, int]:
    while text[start].isspace():
        start += 1
    while text[end - 1].isspace():
        end -= 1
    return start, end, n


def _pack(pieces: Iterator[tuple[int, int, int]], max_tokens: int) -> Iterator[tuple[int, int, int]]:
    """Join consecutive pieces greedily while the total fits; what lies between pieces is whitespace."""
    current: tuple[int, int, int] | None = None
    for start, end, n in pieces:
        if current is not None and current[2] + n <= max_tokens:
            current = (current[0], end, current[2] + n)
            continue
        if current is not None:
            yield current
        current = (start, end, n)
    if current is not None:
        yield current
