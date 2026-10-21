"""Properties of normalization over random pages built from known parts, so the test knows what each part must
become: code untouched, includes expanded once each, no build syntax left outside code, idempotent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hypothesis import given, settings
from hypothesis import strategies as st

from recall import markdown
from recall.corpus.normalize import normalize

FILES = {
    "docs_src/a.py": "from fastapi import FastAPI\n\napp = FastAPI()\n# a comment\n",
    "docs_src/b.py": 'x = "<b>not a tag</b>"\n\n\ny = 2\n',
    "docs_src/t.html": "<p>{{ name }}</p>\n",
}
TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")


def read(path: str) -> str:
    return FILES[path]


@dataclass
class Page:
    parts: list[str] = field(default_factory=list)
    fences: list[str] = field(default_factory=list)  # the page's own fenced blocks, which must come out unchanged
    words: list[str] = field(default_factory=list)  # visible words of text lines, which must survive
    code_includes: int = 0


WORD = st.sampled_from(["query", "Depends", "item_id", "a.b", "42", "naïve", "(x)"])


@st.composite
def text_line(draw: st.DrawFn, page: Page) -> str:
    words = draw(st.lists(WORD, min_size=1, max_size=6))
    page.words.extend(words)
    out = []
    for w in words:
        wrap = draw(st.sampled_from(["", "", "dfn", "abbr", "code-span", "comment"]))
        if wrap in ("dfn", "abbr"):
            out.append(f'<{wrap} title="t">{w}</{wrap}>')
        elif wrap == "code-span":
            out.append(f"`{w} <b>`")
        elif wrap == "comment":
            out.append(f"{w}<!-- gone -->")
        else:
            out.append(w)
    return "t " + " ".join(out)


@st.composite
def pages(draw: st.DrawFn) -> Page:
    page = Page()
    for _ in range(draw(st.integers(0, 10))):
        kind = draw(st.sampled_from(["text", "code-include", "raw-include", "fence", "admonition", "tab", "comment"]))
        if kind == "text":
            page.parts.append("\n".join(draw(st.lists(text_line(page), min_size=1, max_size=3))))
        elif kind == "code-include":
            path = draw(st.sampled_from(["docs_src/a.py", "docs_src/b.py"]))
            opts = draw(st.sampled_from(["", " hl[1]", " ln[1:2] hl[2]", ' title["x.py"]']))
            indent = draw(st.sampled_from(["", "    "]))
            page.parts.append(f"{indent}{{* ../../{path}{opts} *}}")
            page.code_includes += 1
        elif kind == "raw-include":
            page.parts.append('```jinja hl_lines="1"\n{!../../docs_src/t.html!}\n```')
        elif kind == "fence":
            body = draw(
                st.lists(
                    st.sampled_from(
                        ["# a heading? no", "/// tip", "{* ../../docs_src/a.py *}", "<b>x</b>", "", "y = 1"]
                    ),
                    max_size=4,
                )
            )
            fence = "\n".join(["```python", *body, "```"])
            page.fences.append(fence)
            page.parts.append(fence)
        elif kind == "admonition":
            name = draw(st.sampled_from(["tip", "note", "warning", "details"]))
            title = draw(st.sampled_from(["", " | Technical Details", " | Uses `<b>`"]))
            page.parts.append(f"/// {name}{title}\n\n{draw(text_line(page))}\n\n///")
        elif kind == "tab":
            label = draw(st.sampled_from(["Python 3.10+", "`uvicorn`"]))
            page.parts.append(f"//// tab | {label}\n\n{draw(text_line(page))}\n\n////")
        else:
            page.parts.append("<!--\nhidden\n-->")
    return page


def fenced_blocks(text: str) -> list[str]:
    blocks: dict[int, list[str]] = {}
    for ln in markdown.scan(text):
        if ln.fence is not None:
            blocks.setdefault(ln.fence, []).append(text[ln.start : ln.end])
    return ["\n".join(lines) for lines in blocks.values()]


def outside_fences(text: str) -> list[str]:
    return [text[ln.start : ln.end] for ln in markdown.scan(text) if ln.fence is None]


@settings(max_examples=300, deadline=None)
@given(page=pages(), separator=st.sampled_from(["\n\n", "\n\n\n"]))
def test_normalization_properties(page: Page, separator: str):
    source = separator.join(page.parts) + "\n"
    out = normalize("p.md", source, read)

    # The page's own code comes out unchanged, in order, and each code include adds exactly one block.
    blocks = fenced_blocks(out)
    own = [b for b in blocks if b in page.fences]
    assert own == page.fences
    raw_includes = sum(1 for p in page.parts if "{!" in p)
    assert len(blocks) == len(page.fences) + page.code_includes + raw_includes

    # Nothing of the documentation build is left outside code, and neither are tags or comments.
    for line in outside_fences(out):
        assert not line.lstrip().startswith(("{*", "{!", "///")), line
        assert "<!--" not in line and "-->" not in line, line
        assert not TAG.search(CODE_SPAN.sub("", line)), line
        assert "hidden" not in line

    # Every visible word of the text survives.
    visible = " ".join(outside_fences(out))
    for w in page.words:
        assert w in visible

    # Normalizing again changes nothing.
    assert normalize("p.md", out, read) == out
