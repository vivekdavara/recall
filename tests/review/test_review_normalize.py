"""Review findings for recall.corpus.normalize (DESIGN.md "Normalization"). Each test fails on commit 40a7144."""

from __future__ import annotations

from recall import chunking
from recall.corpus.normalize import normalize

FILES = {
    "docs_src/a.py": "x = 1\n",
    "docs_src/t/item.html": "<html>\n<b>{{ id }}</b>\n</html>\n",
}


def read(path: str) -> str:
    try:
        return FILES[path]
    except KeyError:
        raise FileNotFoundError(path) from None


def norm(text: str) -> str:
    return normalize("tutorial/page.md", text, read)


def test_a_comment_around_a_fenced_block_is_removed_with_it():
    """An HTML comment that comments out a code example must vanish entirely ("HTML comments are removed").

    Today the scanner marks the fence lines inside the comment as code before comments are considered, and
    `normalize` copies fenced lines verbatim, so the commented-out example leaks into the stored document.
    """
    page = "Intro.\n\n<!--\n```python\nold_api()\n```\n-->\n\nAfter.\n"
    out = norm(page)
    assert "old_api" not in out, out
    assert "```" not in out, out


def test_a_comment_holding_only_an_opening_fence_does_not_turn_the_rest_of_the_page_into_code():
    """A comment containing a lone fence line (e.g. a half-commented example) must not leak that fence.

    Today the "```" line (and the closing "-->") are copied out, the output now has an unclosed fence, and the
    chunker treats every later heading as code: "# Next" below gets no section of its own (C5).
    """
    page = "<!--\n```\n-->\n\n# Next { #next }\n\nBody.\n"
    out = norm(page)
    assert "-->" not in out, out
    chunks = chunking.chunk_document(out)
    assert [c.heading_path for c in chunks if "Body." in c.text] == [("Next",)], chunks


def test_comment_syntax_inside_inline_code_is_kept_and_does_not_swallow_the_page():
    """Inline code is left untouched (DESIGN.md rule 3), so `<!--` in a code span is text, not a comment opener.

    Today `_strip_comments` runs before code spans are considered: everything after `<!--` up to the next `-->`
    anywhere later in the page is silently dropped, headings and all.
    """
    page = "Start a comment with `<!--`.\n\n# Next { #next }\n\nBody text.\n\nEnd it with `-->`.\n"
    out = norm(page)
    assert "`<!--`" in out, out
    assert "Body text." in out and "# Next { #next }" in out, out


def test_a_closed_comment_inside_inline_code_is_kept():
    """`` `<!-- x -->` `` is an inline code span and must survive, like `` `<b>` `` does. Today it becomes ``."""
    assert norm("Use `<!-- note -->` in a template.") == "Use `<!-- note -->` in a template."


def test_a_raw_include_inside_a_comment_is_not_expanded():
    """A commented-out include is part of the comment and must be removed with it, not expanded into the page."""
    out = norm("<!--\n{!../../docs_src/a.py!}\n-->\nafter")
    assert "x = 1" not in out, out


def test_a_raw_include_outside_a_fence_is_read_as_page_text():
    """Changed from the review's version, which expected a fenced block because DESIGN.md said so. The build's
    markdown-include inserts the file's text as Markdown, so the rendered page shows it as page text: Recall now
    inserts it and normalizes it with the page (its tags go), which also makes the next test pass. DESIGN.md now
    says this."""
    out = norm("Template:\n\n{!../../docs_src/t/item.html!}\n\nAfter.")
    assert out == "Template:\n\n\n{{ id }}\n\n\nAfter.", out


def test_normalization_is_idempotent_with_a_raw_include_outside_a_fence():
    """DESIGN.md: "normalizing its own output changes nothing". The unfenced include's tags go on a second pass."""
    once = norm("Template:\n\n{!../../docs_src/t/item.html!}\n\nAfter.")
    assert norm(once) == once


def test_a_tag_inside_a_code_span_that_wraps_onto_the_next_line_stays():
    """A code span may continue onto the next line (CommonMark); its `<b>` is code and must be left untouched.

    Today tags are stripped line by line, so a hard-wrapped code span loses its tag text.
    """
    page = "Wrap the word in `<b>\nbold</b>` tags."
    assert norm(page) == page


def test_the_text_of_a_style_element_is_not_kept_as_page_text():
    """The normalized page should read the way the rendered page does; CSS inside <style> is never shown.

    FastAPI's index.md has exactly this block; today its CSS rule becomes the first words of a chunk.
    """
    page = "# FastAPI\n\n<style>\n.md-content .md-typeset h1 { display: none; }\n</style>\n\nFastAPI is fast.\n"
    out = norm(page)
    assert "display: none" not in out, out
