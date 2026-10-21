"""Review 2 findings for recall.corpus.normalize (DESIGN.md "Normalization"). Each test fails on commit 0e87c41."""

from __future__ import annotations

import time

import pytest

from recall import chunking, markdown
from recall.corpus.normalize import normalize

FILES = {
    "docs_src/inc.md": "Included text.",
    "docs_src/open_span.md": "Use `<b>",
}


def read(path: str) -> str:
    try:
        return FILES[path]
    except KeyError:
        raise FileNotFoundError(path) from None


def norm(text: str) -> str:
    return normalize("tutorial/page.md", text, read)


def heading_paths(text: str) -> list[tuple[str, ...]]:
    return [s.heading_path for s in chunking.sections(text)]


@pytest.mark.parametrize(
    "page",
    [
        # A paragraph (an HTML block needs only whitespace after its tag): the backticks are literal text.
        "<b></b>```\n\n# Real heading\n\ntext",
        # An HTML block (a line starting with <!--) ends with its line: the backticks are part of it.
        "<!-- note -->```\n\n# Real heading\n\ntext",
        # "What follows on that line is page text": text, not the start of a fenced block.
        "<!--\ncomment\n-->```\n\n# Real heading\n\ntext",
    ],
    ids=["after-a-tag", "after-a-closed-comment", "after-a-comment-closes"],
)
def test_removing_markup_does_not_invent_a_fence_that_hides_the_rest_of_the_page(page: str):
    """The page has no fenced block, so the normalized text must not have one either: today the removed markup
    leaves "```" at the start of a line, which opens a fence in the normalized document, and the real heading below
    it (and everything to the end) becomes code to the chunker. DESIGN.md: normalization turns the page into plain
    Markdown "that reads the way the rendered page does"; the rendered page shows "```" as text and has the heading.
    """
    out = norm(page)
    assert all(markdown.fence_opener(line) is None for line in out.split("\n")), out
    assert ("Real heading",) in heading_paths(out)


def test_removing_a_tag_does_not_invent_a_heading():
    """'<b>#</b> Title' is a paragraph, not a heading (normalize's own _strip_closed_comments docstring says so),
    but the normalized line is '# Title', which the chunker reads as a heading: a section that isn't on the page,
    and "Title" leaves the chunk text (heading lines aren't chunk text, C3)."""
    out = norm("<b>#</b> Title\n\nBody.")
    assert ("Title",) not in heading_paths(out), out
    assert any("Title" in c.text for c in chunking.chunk_document(out)), out


@pytest.mark.parametrize(
    "page",
    [
        "/// tip | <b>Bold</b> title\n\nBody.\n\n///",
        "/// tip | Title <!-- note\n\nhidden\n\n-->\n\nvisible",
        "<b></b>/// tip\n\nBody.",
        "<b></b>{!../../docs_src/inc.md!}\n\nBody.",
        "---\na: 1\n---\n---\nb: 2\n---\nBody.",
    ],
    ids=[
        "tag-in-a-marker-label",
        "comment-in-a-marker-label",
        "tag-before-a-marker",
        "tag-before-an-include",
        "front-matter-like-block-after-front-matter",
    ],
)
def test_normalization_is_idempotent(page: str):
    """DESIGN.md: "normalizing its own output changes nothing ... except for an include line inside an included
    file". Today a marker's label keeps its tags and comment openers (the second pass removes them, and with an
    unclosed comment drops the lines after it); markup before a marker or an include leaves the bare marker or
    include line, which the second pass turns into a label or expands; and a second '---' block at the top of the
    output is taken for front matter and removed by the second pass."""
    once = norm(page)
    assert norm(once) == once


def test_a_marker_label_loses_its_tags():
    """Rule 4: tags are removed and the text between them kept. The admonition title is rendered as inline
    Markdown, so '<b>Bold</b> title' reads 'Bold title'; today the label line keeps '<b>' and '</b>'."""
    assert norm("/// tip | <b>Bold</b> title\n\nBody.\n\n///").split("\n")[0] == "Tip: Bold title"


def test_an_element_whose_name_starts_with_style_is_not_a_style_element():
    """Only <style> and <script> elements are dropped with their content. '<style-guide>' is another element
    (a custom element name), whose text the page shows; today `<style\\b` matches it, and everything up to a
    '</style>' that never comes (here: the rest of the page) is dropped."""
    out = norm("See <style-guide>the guide</style-guide> here.\n\nNext paragraph.")
    assert "the guide" in out and "here." in out and "Next paragraph." in out, out


def test_a_code_span_may_continue_from_an_included_file_onto_the_page():
    """A raw include outside a fence "inserts the file's text, which is then read as the page's own text", and code
    spans are "found over the whole paragraph". The included 'Use `<b>' and the page's next line '</b>` tags' are
    one paragraph whose code span holds '<b>' and '</b>'; today the paragraph is cut at the include's end, so both
    tags are removed as markup."""
    out = norm("{!../../docs_src/open_span.md!}\n</b>` tags")
    assert out == "Use `<b>\n</b>` tags"


def test_many_unclosed_style_openers_take_linear_time():
    """Normalizing should scale with the page's length. Each '<style' without a '>' after it makes _RAW_ELEMENT's
    [^>]* scan to the end of the paragraph, so a paragraph of n such lines takes time quadratic in n (about 0.05 s
    for 4,000 lines, 13 s projected for 64,000 here). 20,000 lines are 0.2 MB and take seconds today."""
    page = "\n".join("a <style b" for _ in range(20_000))
    start = time.perf_counter()
    norm(page)
    assert time.perf_counter() - start < 0.5
