"""Turn a FastAPI documentation page into plain Markdown that reads like the rendered page.

DESIGN.md "Normalization" is the specification. One pass over the lines: fenced code is kept (raw includes, which
FastAPI puts inside fences, are expanded there); HTML comments and <style>/<script> elements are dropped; in page
text, code includes become fenced blocks, raw includes are read as page text, admonitions and tabs become label
lines, and comments and tags go outside code spans. Heading ids stay in their heading lines: they are the chunker's
anchors.
"""

from __future__ import annotations

import posixpath
import re
from collections import deque
from collections.abc import Callable

from recall import markdown

# Includes are written relative to the documentation build's root, docs/en/ in the repository.
DOCS_ROOT = "docs/en"

# {* ../../docs_src/x.py ln[1:3,8] hl[2] title[...] *} on a line of its own: a code block of a Python file.
_CODE_INCLUDE = re.compile(r"^([ \t]*)\{\*[ \t]+(\S+)((?:[ \t]+[a-z]+\[[^\]]*\])*)[ \t]*\*\}[ \t]*$")
_OPTION = re.compile(r"([a-z]+)\[([^\]]*)\]")
# {!path!} or {!> path!} on a line of its own: the file's text, as is.
_RAW_INCLUDE = re.compile(r"^([ \t]*)\{!>?[ \t]*(\S+?)[ \t]*!\}[ \t]*$")
# /// tip | Title, //// tab | Python 3.10+, and the closing /// or ////.
_BLOCK_MARKER = re.compile(r"^([ \t]*)/{3,}(?:[ \t]+([A-Za-z]+))?(?:[ \t]*\|[ \t]*(.*?))?[ \t]*$")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")

_LANGUAGES = {".py": "python", ".html": "jinja", ".css": "css", ".js": "javascript", ".toml": "toml", ".sh": "bash"}

Reader = Callable[[str], str]


class NormalizeError(ValueError):
    pass


def include_path(target: str) -> str:
    """The repository path an include refers to: relative to DOCS_ROOT (not to the page), inside the repository."""
    resolved = posixpath.normpath(posixpath.join(DOCS_ROOT, target))
    if resolved.startswith("../") or resolved == ".." or posixpath.isabs(resolved):
        raise NormalizeError(f"include outside the repository: {target}")
    return resolved


def includes(page_path: str, text: str, read: Reader) -> list[str]:
    """Every repository path the page includes, in order of first use: exactly the files normalize reads."""
    found: list[str] = []

    def record(path: str) -> str:
        if path not in found:
            found.append(path)
        return read(path)

    normalize(page_path, text, record)
    return found


def normalize(page_path: str, text: str, read: Reader) -> str:
    """The page as plain Markdown. `read` returns an included file's text by its repository path.

    One pass over the lines, in order, in one of three states: inside a fenced code block (code: kept as is, but
    a raw include is expanded), inside an HTML comment or a <style>/<script> element (dropped, up to its end), or
    in page text. Whichever starts first wins, as in CommonMark: a fence inside a comment is part of the comment,
    and a comment inside a fence is code.
    """
    if "\x00" in text:
        raise NormalizeError(f"{page_path}: NUL character")
    text = _strip_front_matter(_prepare(text))
    out: list[str] = []
    # (line, included): lines a raw include brings in are read as the page's own text, but not searched for
    # further includes (one level, as the documentation build does).
    queue: deque[tuple[str, bool]] = deque((line, False) for line in text.split("\n"))
    fence: tuple[str, int] | None = None
    closer: re.Pattern[str] | None = None
    while queue:
        line, included = queue.popleft()
        if fence is not None:
            raw = None if included else _RAW_INCLUDE.match(line)
            if raw:
                out.extend(_raw_include(page_path, raw.group(1), raw.group(2), read, inside_fence=True))
                continue
            out.append(line)
            if markdown.closes(line, fence):
                fence = None
            continue
        if closer is not None:
            m = closer.search(line)
            if m is None:
                out.append("")
                continue
            rest, closer = _inline(line[m.end() :])
            out.append(_escape(rest))
            continue
        probe = _strip_closed_comments(line)
        opener = markdown.fence_opener(line)
        if opener is not None:
            fence = opener
            out.append(line)
            continue
        if not included:
            raw = _RAW_INCLUDE.match(probe)
            if raw:
                lines = _raw_include(page_path, raw.group(1), raw.group(2), read, inside_fence=False)
                queue.extendleft((ln, True) for ln in reversed(lines))
                continue
            code = _CODE_INCLUDE.match(probe)
            if code:
                out.extend(_code_include(page_path, code.group(1), code.group(2), code.group(3), read))
                continue
        marker = _BLOCK_MARKER.match(probe)
        if marker:
            label, closer = _inline(_label(marker.group(1), marker.group(2), marker.group(3)))
            out.append(_escape(label))
            continue
        if markdown.is_blank(line) or markdown.heading(probe) is not None:
            processed, closer = _inline(line)
            out.append(processed)
            continue
        # A paragraph: this line and the plain text lines after it, so that a code span may wrap onto the next line.
        # Included lines and the page's own join one paragraph: the included text is read as the page's, so a
        # raw include met here is expanded in place and its lines may continue the paragraph.
        paragraph = [line]
        while queue:
            nxt, nxt_included = queue[0]
            raw = None if nxt_included else _RAW_INCLUDE.match(_strip_closed_comments(nxt))
            if raw:
                queue.popleft()
                lines = _raw_include(page_path, raw.group(1), raw.group(2), read, inside_fence=False)
                queue.extendleft((ln, True) for ln in reversed(lines))
                continue
            if not _is_paragraph_line(nxt, nxt_included):
                break
            paragraph.append(queue.popleft()[0])
        processed, closer = _inline("\n".join(paragraph))
        out.extend(_escape(ln) for ln in processed.split("\n"))
    result = "\n".join(out)
    # Output that starts with something front matter-shaped (a rule after the real front matter, markup that
    # became '---', an included file's first lines) would lose it on a second pass: a blank line first keeps it.
    return "\n" + result if _strip_front_matter(result) != result else result


def _escape(line: str) -> str:
    """A line of page text that reads as block syntax only because markup was removed from it gets a backslash.

    '<b>#</b> Title' becomes '# Title' and '<!-- x -->```' becomes '```' once their markup goes: the rendered page
    shows that text, not a heading or a fence. CommonMark's backslash escape keeps it text here and on every later
    pass, so normalizing again changes nothing. A text line can only be block syntax if markup was removed (or it
    is an include line inside an included file, which is not expanded: one level).
    """
    if not _is_block_line(line):
        return line
    indent = len(line) - len(line.lstrip(" \t"))
    return line[:indent] + "\\" + line[indent:]


def _is_block_line(line: str) -> bool:
    return (
        markdown.fence_opener(line) is not None
        or markdown.heading(line) is not None
        or bool(_BLOCK_MARKER.match(line) or _RAW_INCLUDE.match(line) or _CODE_INCLUDE.match(line))
    )


def _is_paragraph_line(line: str, included: bool) -> bool:
    """Whether a line continues a paragraph: not blank, and not a fence, heading, marker or include line."""
    if markdown.is_blank(line) or markdown.fence_opener(line) is not None:
        return False
    probe = _strip_closed_comments(line)
    if markdown.heading(probe) is not None or _BLOCK_MARKER.match(probe):
        return False
    return included or not (_RAW_INCLUDE.match(probe) or _CODE_INCLUDE.match(probe))


def title(text: str, fallback: str) -> str:
    """The page's first level-1 heading outside fenced code, or the fallback."""
    for line, fenced in _lines(text):
        if not fenced:
            h = markdown.heading(line)
            if h is not None and h.level == 1 and h.title:
                return h.title
    return fallback


def _strip_front_matter(text: str) -> str:
    """Remove YAML front matter: a first line '---', up to the next line that is '---' or '...'.

    The documentation build reads it as page settings (index.md's sponsors file, templates.md's render_macros) and
    doesn't show it. Without a closing line it isn't front matter, and a '---' later in a page is a horizontal rule.
    """
    if not text.startswith("---\n"):
        return text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i] in ("---", "..."):
            return "\n".join(lines[i + 1 :])
    return text


def _lines(text: str) -> list[tuple[str, bool]]:
    return [(text[ln.start : ln.end], ln.fence is not None) for ln in markdown.scan(text)]


def _read(page_path: str, target: str, read: Reader) -> str:
    path = include_path(target)
    try:
        content = read(path)
    except FileNotFoundError as e:
        raise NormalizeError(f"{page_path}: included file not found: {path}") from e
    if "\x00" in content:
        raise NormalizeError(f"{page_path}: NUL character in {path}")
    return content.replace("\r\n", "\n").removesuffix("\n")


def _code_include(page_path: str, indent: str, target: str, options: str, read: Reader) -> list[str]:
    lines = _read(page_path, target, read).split("\n")
    selected = lines
    for name, value in _OPTION.findall(options):
        if name == "ln":
            selected = _select_lines(page_path, lines, value)
    fence = "`" * max(3, _longest_run(selected, "`") + 1)
    language = _LANGUAGES.get(posixpath.splitext(target)[1], "")
    body = [indent + ln if ln else ln for ln in selected]
    return [indent + fence + language, *body, indent + fence]


def _select_lines(page_path: str, lines: list[str], spec: str) -> list[str]:
    """ln[1:2,12:16,29]: 1-based, inclusive ranges, in order; a line '...' marks code left out between them."""
    picked: list[str] = []
    last = 0
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?::(\d+))?", part)
        if not m:
            raise NormalizeError(f"{page_path}: bad line range {part!r} in ln[{spec}]")
        first, end = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= first <= end <= len(lines):
            raise NormalizeError(f"{page_path}: line range {part} outside 1..{len(lines)}")
        if first <= last:
            raise NormalizeError(f"{page_path}: line ranges out of order in ln[{spec}]")
        if picked and first > last + 1:
            picked.append("...")
        picked.extend(lines[first - 1 : end])
        last = end
    return picked


def _raw_include(page_path: str, indent: str, target: str, read: Reader, inside_fence: bool) -> list[str]:
    lines = _read(page_path, target, read).split("\n")
    if inside_fence:
        # A line of only backticks or only tildes could close the fence the include sits in; refuse rather than
        # let an included file change the page's structure.
        # An include line in it would stay as code here, and be expanded by a second pass: refused too.
        for ln in lines:
            stripped = ln.strip()
            if len(stripped) >= 3 and set(stripped) in ({"`"}, {"~"}):
                raise NormalizeError(f"{page_path}: included {target} has a line that could close its fence")
            if _RAW_INCLUDE.match(ln):
                raise NormalizeError(f"{page_path}: included {target} has an include line inside a fence")
    return [indent + ln if ln else ln for ln in lines]


def _longest_run(lines: list[str], char: str) -> int:
    runs = [len(m.group()) for ln in lines for m in re.finditer(re.escape(char) + "+", ln)]
    return max(runs, default=0)


def _label(indent: str, kind: str | None, label: str | None) -> str:
    """The line an admonition or tab marker becomes; a closing marker becomes a blank line, so blocks stay apart."""
    if kind is None:
        return ""
    label = (label or "").strip().strip('"')
    if kind.lower() == "tab":
        return f"{indent}{label}:"
    name = kind.capitalize()
    return f"{indent}{name}: {label}" if label else f"{indent}{name}:"


def _prepare(text: str) -> str:
    """Line ends and the byte order mark, as the documentation build reads pages (utf-8-sig)."""
    return text.removeprefix("﻿").replace("\r\n", "\n")


_CLOSED_COMMENT = re.compile(r"<!--(?:-?>|.*?-->)")


def _strip_closed_comments(line: str) -> str:
    """The line without the HTML comments that open and close on it, to recognise the block syntax around them.

    Only comments: removing tags here could make syntax that isn't there ('<b>#</b> Title' is not a heading). A
    line with backticks is left as it is, since its `<!--` may be code.
    """
    return _CLOSED_COMMENT.sub("", line) if "<!--" in line and "`" not in line else line


_SPECIAL = re.compile(r"[`<]")
_COMMENT_END = re.compile(r"-->")
# The element's name must end there (<style-guide> is another element). Its end tag is searched for straight from
# the name, whatever its attributes hold (CommonMark's HTML block 1): one search, so no rescanning.
_RAW_ELEMENT = re.compile(r"<(style|script)(?=[\s/>]|$)", re.IGNORECASE)
_BACKTICKS = re.compile(r"`+")


def _inline(text: str) -> tuple[str, re.Pattern[str] | None]:
    """Page text with HTML comments, tags, and <style>/<script> elements removed, code spans left as they are.

    Newlines inside what is removed are kept, so the text keeps its lines. If a comment or element is still open
    at the end, returns the pattern that will close it; the caller drops lines until it matches.
    """
    out: list[str] = []
    i = 0
    while True:
        m = _SPECIAL.search(text, i)
        if m is None:
            out.append(text[i:])
            return "".join(out), None
        j = m.start()
        out.append(text[i:j])
        if text[j] == "`":
            backticks = _BACKTICKS.match(text, j)
            assert backticks is not None  # text[j] is a backtick
            run = backticks.end() - j
            close = re.compile(rf"(?<!`)`{{{run}}}(?!`)").search(text, j + run)
            # A code span: a run of backticks up to the next run of the same length (CommonMark 6.1); without
            # one, the backticks are literal.
            end = close.end() if close else j + run
            out.append(text[j:end])
            i = end
            continue
        if text.startswith("<!--", j):
            # <!--> and <!---> are complete (empty) comments too (CommonMark 0.31), hence the search from j + 2.
            close = _COMMENT_END.search(text, j + 2)
            if close is None:
                out.append("\n" * text.count("\n", j))
                return "".join(out), _COMMENT_END
            out.append("\n" * text.count("\n", j, close.end()))
            i = close.end()
            continue
        element = _RAW_ELEMENT.match(text, j)
        if element:
            end_tag = re.compile(rf"</{element.group(1)}\s*>", re.IGNORECASE)
            close = end_tag.search(text, element.end())
            if close is None:
                out.append("\n" * text.count("\n", j))
                return "".join(out), end_tag
            out.append("\n" * text.count("\n", j, close.end()))
            i = close.end()
            continue
        tag = _TAG.match(text, j)
        if tag:
            out.append("\n" * tag.group().count("\n"))
            i = tag.end()
            continue
        out.append("<")
        i = j + 1
