from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recall.chunking import Chunk, chunk_document, count_tokens, sections


def texts(chunks: list[Chunk]) -> list[str]:
    return [c.text for c in chunks]


# Examples, one rule each.


def test_tokens_are_word_runs_and_single_punctuation():
    assert count_tokens('@app.get("/items/{item_id}")') == 14
    assert count_tokens("  hello   world\n") == 2
    assert count_tokens("") == 0 and count_tokens(" \n\t") == 0
    assert count_tokens("naïve café_au_lait") == 2


def test_empty_and_blank_documents_have_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("\n \n\t\n") == []
    assert chunk_document("# Only a heading { #only }\n## And another\n") == []


def test_heading_lines_are_not_chunk_text_but_give_the_path_and_anchor():
    doc = "Intro.\n\n# Query { #query }\n\nBody one.\n\n## Defaults { #defaults }\n\nBody two.\n"
    chunks = chunk_document(doc)
    assert texts(chunks) == ["Intro.", "Body one.", "Body two."]
    assert [c.heading_path for c in chunks] == [(), ("Query",), ("Query", "Defaults")]
    assert [c.anchor for c in chunks] == [None, "query", "defaults"]


def test_a_heading_closes_deeper_and_equal_levels():
    doc = "# A\n## B\n### C\nc\n## D\nd\n# E\ne\n"
    assert [c.heading_path for c in chunk_document(doc)] == [("A", "B", "C"), ("A", "D"), ("E",)]


def test_a_hash_line_inside_a_fence_is_code_not_a_heading():
    doc = "# Real\n\n```python\n# a comment\nx = 1\n```\n"
    [chunk] = chunk_document(doc)
    assert chunk.heading_path == ("Real",)
    assert chunk.text == "```python\n# a comment\nx = 1\n```"


def test_small_blocks_share_a_chunk_until_the_next_would_not_fit():
    doc = "one two three\n\nfour five\n\nsix seven eight"
    assert texts(chunk_document(doc, max_tokens=5)) == ["one two three\n\nfour five", "six seven eight"]


def test_chunks_never_span_two_sections():
    doc = "# A\na\n# B\nb\n"
    assert texts(chunk_document(doc, max_tokens=100)) == ["a", "b"]


def test_a_fence_that_fits_is_never_split():
    fence = "```\nx = 1\n\ny = 2\n```"
    doc = f"alpha beta\n\n{fence}\n\ngamma"
    # The fence has 12 tokens (each backtick is one); with room for 12 it must be one chunk of its own.
    assert count_tokens(fence) == 12
    assert texts(chunk_document(doc, max_tokens=12)) == ["alpha beta", fence, "gamma"]


def test_a_fence_that_is_too_big_splits_at_line_ends():
    doc = "```\naa bb\ncc dd\nee ff\n```"
    assert texts(chunk_document(doc, max_tokens=5)) == ["```\naa bb", "cc dd\nee ff", "```"]


def test_a_paragraph_before_a_fence_without_a_blank_line_is_its_own_block():
    doc = "Look:\n```\ncode here\n```"  # 2 tokens, then a fence of 8
    assert texts(chunk_document(doc, max_tokens=10)) == ["Look:\n```\ncode here\n```"]
    assert texts(chunk_document(doc, max_tokens=9)) == ["Look:", "```\ncode here\n```"]


def test_a_long_line_splits_at_spaces():
    doc = "a b c d e f g"
    assert texts(chunk_document(doc, max_tokens=3)) == ["a b c", "d e f", "g"]


def test_a_run_too_long_for_a_chunk_splits_between_tokens():
    # The line splits into x, then a . b . c . d one token each, then y; packing joins them greedily.
    doc = "x " + "a.b.c.d" + " y"
    assert texts(chunk_document(doc, max_tokens=3)) == ["x a.", "b.c", ".d y"]


def test_chunks_start_and_end_on_text():
    doc = "   indented first\n\n  * item  \n"
    assert texts(chunk_document(doc)) == ["indented first\n\n  * item"]


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        chunk_document("x", max_tokens=0)


def test_sections_cover_the_document_between_headings():
    doc = "pre\n# A\nbody\n"
    assert [(s.level, s.heading_path, doc[s.start : s.end]) for s in sections(doc)] == [
        (0, (), "pre\n"),
        (1, ("A",), "body\n"),
    ]


# Properties: random documents built from known parts, so the test knows where every heading and fence is.

WORDS = st.sampled_from(
    ["fastapi", "query", "item_id", "x", "Depends", "a.b", "(", ")", "=", "42", "naïve", "é", "->", "{x}", "!"]
)


@dataclass
class Doc:
    text: str = ""
    headings: list[tuple[int, int]] = field(default_factory=list)  # spans of heading lines
    fences: list[tuple[int, int]] = field(default_factory=list)  # spans of fenced blocks, fences included
    paths: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)  # (offset, heading path from there)

    def add(self, s: str) -> tuple[int, int]:
        start = len(self.text)
        self.text += s
        return start, len(self.text)


@st.composite
def line_of_words(draw: st.DrawFn, max_words: int = 12) -> str:
    # Starting with a letter keeps a text line from being read as a heading, a fence or a list marker.
    words = draw(st.lists(WORDS, min_size=0, max_size=max_words))
    long_run = draw(st.sampled_from(["", "", "", "a.b.c.d.e.f.g.h.i.j.k.l", "w" * 50]))
    spaces = draw(st.sampled_from([" ", "  ", "\t"]))
    return spaces.join(["t", *words, long_run]).rstrip()


@st.composite
def documents(draw: st.DrawFn) -> Doc:
    doc = Doc()
    stack: list[tuple[int, str]] = []
    doc.paths.append((0, ()))
    for _ in range(draw(st.integers(0, 12))):
        kind = draw(st.sampled_from(["heading", "paragraph", "paragraph", "fence", "list"]))
        if kind == "heading":
            level = draw(st.integers(1, 6))
            title = draw(st.sampled_from(["Intro", "Query Params", "C#", "`Depends`", ""]))
            anchor = draw(st.sampled_from(["", " { #intro }", " {#q-1}"]))
            span = doc.add(("#" * level + " " + title + anchor).rstrip())
            doc.headings.append(span)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            doc.paths.append((span[1], tuple(t for _, t in stack)))
        elif kind == "paragraph":
            lines = draw(st.lists(line_of_words(), min_size=1, max_size=4))
            doc.add("\n".join(lines))
        elif kind == "list":
            items = draw(st.lists(line_of_words(4), min_size=1, max_size=4))
            indent = draw(st.sampled_from(["", "  "]))
            doc.add("\n".join(f"{indent}* {item}" for item in items))
        else:
            char = draw(st.sampled_from(["`", "~"]))
            n = draw(st.integers(3, 5))
            body_line = st.one_of(
                line_of_words(),
                st.just(""),
                st.just("   "),
                st.just("# not a heading"),
                st.just("## nor this { #no }"),
                st.just(char * (n - 1)),  # too short to close
                st.just(("~" if char == "`" else "`") * 4),  # the other character doesn't close
            )
            body = draw(st.lists(body_line, max_size=6))
            indent = draw(st.sampled_from(["", "    "]))
            info = draw(st.sampled_from(["", "python", " toml"]))
            fence_text = "\n".join([indent + char * n + info, *[indent + b for b in body], indent + char * n])
            doc.fences.append(doc.add(fence_text))
        doc.add(draw(st.sampled_from(["\n", "\n\n", "\n \n\n", "\n\t\n"])))
    return doc


def expected_path(doc: Doc, offset: int) -> tuple[str, ...]:
    path: tuple[str, ...] = ()
    for start, p in doc.paths:
        if start <= offset:
            path = p
    return path


@settings(max_examples=400, deadline=None)
@given(doc=documents(), max_tokens=st.integers(1, 40))
def test_invariants_hold_on_random_documents(doc: Doc, max_tokens: int):
    text = doc.text
    chunks = chunk_document(text, max_tokens)

    # C1 verbatim, starting and ending on text.
    for c in chunks:
        assert c.text == text[c.start : c.end]
        assert c.text and not c.text[0].isspace() and not c.text[-1].isspace()

    # C2 order: ordinals count up and spans don't overlap.
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for a, b in pairwise(chunks):
        assert a.end <= b.start

    # C3 coverage: what no chunk holds is whitespace or a heading line, and no chunk holds part of a heading line.
    covered = [False] * len(text)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
        for hs, he in doc.headings:
            assert c.end <= hs or he <= c.start, (c, text[hs:he])
    for i, ch in enumerate(text):
        if not covered[i] and not ch.isspace():
            assert any(hs <= i < he for hs, he in doc.headings), (i, ch, text)

    # C4 size.
    for c in chunks:
        assert 1 <= c.token_count == count_tokens(c.text) <= max_tokens

    # C5 sections: no heading line inside a chunk (above), and the path is the one in force where it starts.
    for c in chunks:
        assert c.heading_path == expected_path(doc, c.start)

    # C6 code: a fence that fits lies inside one chunk (from its first character that isn't indentation).
    for fs, fe in doc.fences:
        fs += len(text[fs:fe]) - len(text[fs:fe].lstrip())
        if count_tokens(text[fs:fe]) <= max_tokens:
            assert any(c.start <= fs and fe <= c.end for c in chunks), (text[fs:fe], chunks)

    # C7 deterministic.
    assert chunk_document(text, max_tokens) == chunks


@settings(max_examples=200, deadline=None)
@given(text=st.text(st.characters(codec="utf-8"), max_size=300), max_tokens=st.integers(1, 20))
def test_any_text_is_chunked_within_the_rules(text: str, max_tokens: int):
    # No structure known here, so only the rules that don't need it: verbatim, ordered, sized, and every
    # non-whitespace character either in a chunk or on a line that is a heading.
    chunks = chunk_document(text, max_tokens)
    covered = [False] * len(text)
    last_end = 0
    for c in chunks:
        assert c.text == text[c.start : c.end] and c.start >= last_end
        assert 1 <= c.token_count == count_tokens(c.text) <= max_tokens
        last_end = c.end
        for i in range(c.start, c.end):
            covered[i] = True
    # The heading lines are exactly what lies between one section's end and the next one's start.
    secs = sections(text)
    gaps = [(a.end, b.start) for a, b in pairwise(secs)]
    for i, ch in enumerate(text):
        if not covered[i] and not ch.isspace():
            assert any(gs <= i < ge for gs, ge in gaps), (i, ch)
