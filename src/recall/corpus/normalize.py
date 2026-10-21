"""Turn a FastAPI documentation page into plain Markdown that reads like the rendered page.

DESIGN.md "Normalization" is the specification. Outside fenced code: code includes become fenced blocks,
admonitions and tabs become label lines, and HTML comments and tags are removed. Raw includes (`{!path!}`) are
expanded everywhere, because FastAPI's pages put them inside fences. Heading lines are left alone: their ids are
the chunker's anchors.
"""

from __future__ import annotations

import posixpath
import re
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
_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")

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


def includes(page_path: str, text: str) -> list[str]:
    """Every repository path the page includes, in order of first use."""
    found: list[str] = []
    for line, fenced in _lines(text):
        target = None
        m = _RAW_INCLUDE.match(line)
        if m:
            target = m.group(2)
        elif not fenced:
            c = _CODE_INCLUDE.match(line)
            if c:
                target = c.group(2)
        if target is not None:
            path = include_path(target)
            if path not in found:
                found.append(path)
    return found


def normalize(page_path: str, text: str, read: Reader) -> str:
    """The page as plain Markdown. `read` returns an included file's text by its repository path."""
    if "\x00" in text:
        raise NormalizeError(f"{page_path}: NUL character")
    text = _strip_front_matter(text.replace("\r\n", "\n"))
    out: list[str] = []
    in_comment = False
    for line, fenced in _lines(text):
        raw = _RAW_INCLUDE.match(line)
        if raw:
            out.extend(_raw_include(page_path, raw.group(1), raw.group(2), read, inside_fence=fenced))
            continue
        if fenced:
            out.append(line)
            continue
        if in_comment:
            end = line.find("-->")
            if end < 0:
                out.append("")
                continue
            line = line[end + 3 :]
            in_comment = False
        line, in_comment = _strip_comments(line)
        code = _CODE_INCLUDE.match(line)
        if code:
            out.extend(_code_include(page_path, code.group(1), code.group(2), code.group(3), read))
            continue
        marker = _BLOCK_MARKER.match(line)
        if marker:
            out.append(_label(marker.group(1), marker.group(2), marker.group(3)))
            continue
        out.append(_strip_tags(line))
    return "\n".join(out)


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
        for ln in lines:
            stripped = ln.strip()
            if len(stripped) >= 3 and set(stripped) in ({"`"}, {"~"}):
                raise NormalizeError(f"{page_path}: included {target} has a line that could close its fence")
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


def _strip_comments(line: str) -> tuple[str, bool]:
    """Remove HTML comments from a line; report whether one is left open at its end."""
    out = ""
    while True:
        start = line.find("<!--")
        if start < 0:
            return out + line, False
        end = line.find("-->", start + 4)
        if end < 0:
            return out + line[:start], True
        out += line[:start]
        line = line[end + 3 :]


def _strip_tags(line: str) -> str:
    """Remove HTML tags outside inline code spans, keeping the text between them."""
    out: list[str] = []
    pos = 0
    for span in _CODE_SPAN.finditer(line):
        out.append(_TAG.sub("", line[pos : span.start()]))
        out.append(span.group())
        pos = span.end()
    out.append(_TAG.sub("", line[pos:]))
    return "".join(out)
