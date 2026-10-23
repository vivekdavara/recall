"""The extractive answer: the passages' own sentences that best answer the question, each citing its passage.

DESIGN.md "Answers" is the specification. Used when no LLM is configured, when the provider fails, and when none of
the model's claims has a citation that checks out. It can't say anything the passages don't, and every citation is
its sentence's own span, so it passes the citation check by construction (and is checked anyway).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from recall import markdown
from recall.citations import Citation, Claim
from recall.rerank import Reranker
from recall.retrieval import Hit

DEFAULT_SENTENCES = 3
# Shorter sentences ("This includes, for example:", an admonition's "Tip:") say too little to be an answer.
MIN_WORDS = 4

# A list item's marker: "* ", "- ", "+ ", "1. " or "1) ", after any indentation.
_LIST_ITEM = re.compile(r"^[ \t]*(?:[*+-]|\d{1,9}[.)])[ \t]+")
# A sentence ends at '.', '!' or '?' (and any closing quotes, brackets, '*' or '`'), then whitespace.
_SENTENCE_END = re.compile(r"[.!?][\"')\]*`]*(?=\s)")
# Words whose period doesn't end a sentence.
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "Mr.", "Dr.")


@dataclass(frozen=True)
class Sentence:
    passage: int  # 1-based
    start: int  # offsets into the passage's content
    end: int
    text: str  # content[start:end]


def sentences(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """The spans of the prose sentences in text[start:end]: outside fenced code, not headings, list markers left out.

    `text` is a chunk, or better its whole document with [start, end) the chunk's span: a code block too long for one
    chunk is split across chunks, so a chunk can start inside one, and only the document shows that its first lines
    are code. A block is a run of prose lines between blank lines; a list item starts a block of its own. Within a
    block a sentence ends after '.', '!' or '?' followed by whitespace, except after a common abbreviation.
    """
    end = len(text) if end is None else end
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    block_end = 0
    for line in markdown.scan(text):
        if line.end < start or line.start >= end:
            continue
        s, e = max(line.start, start), min(line.end, end)
        whole = text[line.start : line.end]
        prose = line.fence is None and not markdown.is_blank(text[s:e]) and markdown.heading(whole) is None
        # A marker counts only on a line that starts inside the span (a cut mid-line is mid-sentence).
        item = _LIST_ITEM.match(whole) if prose and line.start >= start else None
        if block_start is not None and (not prose or item):
            blocks.append((block_start, block_end))
            block_start = None
        if prose:
            if block_start is None:
                block_start = s + (item.end() if item else 0)
            block_end = e
    if block_start is not None:
        blocks.append((block_start, block_end))
    spans: list[tuple[int, int]] = []
    for bs, be in blocks:
        at = bs
        for m in _SENTENCE_END.finditer(text, bs, be):
            if text[at : m.end()].endswith(_ABBREVIATIONS):
                continue
            spans.append((at, m.end()))
            at = m.end()
        spans.append((at, be))
    trimmed = []
    for s, e in spans:
        piece = text[s:e]
        s, e = s + len(piece) - len(piece.lstrip()), e - len(piece) + len(piece.rstrip())
        if s < e:
            trimmed.append((s, e))
    return trimmed


def candidates(passages: Sequence[Hit], documents: Mapping[str, str] | None = None) -> list[Sentence]:
    """Every sentence of every passage with at least MIN_WORDS words, in passage order.

    `documents` maps a page's path to its normalized text, so that code is found from the whole page; a passage
    whose page isn't there (or no longer holds the passage) is read alone.
    """
    found = []
    for number, hit in enumerate(passages, 1):
        document = (documents or {}).get(hit.path)
        if document is not None and document[hit.start : hit.end] == hit.content:
            spans = [(s - hit.start, e - hit.start) for s, e in sentences(document, hit.start, hit.end)]
        else:
            spans = sentences(hit.content)
        for s, e in spans:
            text = hit.content[s:e]
            if len(text.split()) >= MIN_WORDS:
                found.append(Sentence(number, s, e, text))
    return found


def extract(
    question: str,
    passages: Sequence[Hit],
    reranker: Reranker,
    n: int = DEFAULT_SENTENCES,
    documents: Mapping[str, str] | None = None,
) -> list[Claim]:
    """The n sentences the reranker scores highest for the question, best first, each a claim citing its own span.
    Equal scores keep passage order, so the result repeats. `documents` as for candidates()."""
    found = candidates(passages, documents)
    if not found:
        return []
    scores = reranker.score(question, [s.text for s in found])
    if len(scores) != len(found):
        raise ValueError(f"{reranker.name} gave {len(scores)} scores for {len(found)} sentences")
    if not all(math.isfinite(x) for x in scores):
        raise ValueError(f"{reranker.name} gave a score that isn't a finite number")
    best = sorted(range(len(found)), key=lambda i: (-scores[i], i))[:n]
    claims = []
    for i in best:
        s = found[i]
        hit = passages[s.passage - 1]
        citation = Citation(s.passage, hit.chunk_id, hit.path, hit.url, s.text, hit.start + s.start, hit.start + s.end)
        claims.append(Claim(" ".join(s.text.split()), (citation,)))
    return claims
