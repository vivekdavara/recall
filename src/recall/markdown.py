"""The parts of Markdown that Recall's normalization and chunking need: lines, fenced code blocks and ATX headings.

This is deliberately not a full CommonMark parser. Two rules matter for correctness and both follow CommonMark:
a fenced code block runs from its opening fence to a closing fence of the same character that is at least as long
(or to the end of the document), and nothing inside it is a heading. One rule is looser than CommonMark: a fence
may be indented by any amount, because FastAPI's pages indent fences inside list items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# An opening fence: three or more backticks or tildes, then an info string. A backtick fence's info string can't
# contain a backtick (CommonMark 4.5), so "``` x ` y" is a paragraph, not a fence.
_FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")

# An ATX heading: up to three spaces, one to six '#', then a space and the content, or nothing. The content then
# loses its closing sequence as commonmark.js does: content that is one run of '#' is empty ('### ###'), otherwise a
# trailing run of '#' after a space goes ('### ## #' has the title '##').
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?$")
_ONLY_CLOSING = re.compile(r"^[ \t]*#+[ \t]*$")
_CLOSING = re.compile(r"[ \t]+#+[ \t]*$")

# A heading id in attribute-list syntax at the end of a title: "Defaults { #defaults }".
_HEADING_ID = re.compile(r"[ \t]*\{[ \t]*#([A-Za-z0-9_:.-]+)[ \t]*\}$")


@dataclass(frozen=True)
class Line:
    """One line of a document: its span (the newline excluded) and the fenced block it belongs to, if any."""

    start: int
    end: int
    fence: int | None  # index of the fenced code block (opening and closing fences included), or None


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    anchor: str | None


def scan(text: str) -> list[Line]:
    """Split text into lines at '\\n' and mark the lines of each fenced code block."""
    lines: list[Line] = []
    pos = 0
    fence: tuple[str, int] | None = None
    fence_index = -1
    for raw in text.split("\n"):
        start, end = pos, pos + len(raw)
        pos = end + 1
        if fence is None:
            fence = fence_opener(raw)
            if fence is not None:
                fence_index += 1
            lines.append(Line(start, end, fence_index if fence is not None else None))
            continue
        lines.append(Line(start, end, fence_index))
        if closes(raw, fence):
            fence = None
    return lines


def fence_opener(line: str) -> tuple[str, int] | None:
    """The fence character and length if this line opens a fenced code block (outside one), else None."""
    m = _FENCE_OPEN.match(line)
    if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
        return m.group(1)[0], len(m.group(1))
    return None


def closes(line: str, fence: tuple[str, int]) -> bool:
    """Whether this line closes the fenced code block that `fence` (from fence_opener) opened."""
    stripped = line.strip()
    return len(stripped) >= fence[1] and set(stripped) == {fence[0]}


def heading(line: str) -> Heading | None:
    """The heading on this line, if it is one. Callers must not ask about lines inside fenced code."""
    m = _HEADING.match(line)
    if not m:
        return None
    title = _CLOSING.sub("", _ONLY_CLOSING.sub("", m.group(2) or ""))
    anchor = None
    a = _HEADING_ID.search(title)
    if a:
        anchor = a.group(1)
        title = title[: a.start()]
    return Heading(level=len(m.group(1)), title=title.strip(), anchor=anchor)


def is_blank(line: str) -> bool:
    """A line with no characters but whitespace (str.isspace's whitespace, which is also the tokenizer's)."""
    return not line.strip()
