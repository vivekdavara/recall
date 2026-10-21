"""Review 3 findings for the review 2 fixes in recall.corpus.normalize (DESIGN.md "Normalization").

Each test fails on commit 0777cfe."""

from __future__ import annotations

import pytest

from recall.corpus.normalize import NormalizeError, normalize

FILES = {
    "docs_src/fm.md": "---\nIncluded first line\n---\nIncluded rest.",
    "docs_src/q/nested.md": "x = 1\n{!../../docs_src/t/item.html!}\n",
    "docs_src/t/item.html": "<p>{{ item }}</p>",
    "docs_src/tail.md": "b</i>` c",
}


def read(path: str) -> str:
    try:
        return FILES[path]
    except KeyError:
        raise FileNotFoundError(path) from None


def norm(text: str) -> str:
    return normalize("tutorial/page.md", text, read)


@pytest.mark.parametrize(
    "page",
    [
        # Markup removal makes the first line '---': the output now starts like front matter.
        "<b>---</b>\nKept line\n---\nRest.",
        # The same after real front matter: the fix checks the rest before normalizing it, not after.
        "---\ntitle: x\n---\n<!-- c -->---\nKept line\n---\nRest.",
        # An included file is not front matter in the build (only the page's own first line can be), so its '---'
        # lines are page text; once they are the output's first lines, a second pass removes them.
        "{!../../docs_src/fm.md!}",
    ],
    ids=["markup-made-rule", "after-front-matter", "included-file"],
)
def test_output_that_starts_with_a_rule_is_not_front_matter_on_a_second_pass(page: str):
    """DESIGN.md: "Normalization is deterministic and idempotent: normalizing its own output changes nothing", and
    "Front matter followed by another `---` block leaves a blank line first". _strip_front_matter only looks at the
    raw rest of the page, so when the *normalized* output's first line is '---' (from '<b>---</b>', from
    '<!-- c -->---', or from an included file), the second pass strips 'Kept line' as front matter. The output
    should not start with a '---' line that would read as front matter (e.g. prepend the blank line based on the
    output, or escape it)."""
    once = norm(page)
    assert norm(once) == once
    assert "line" in norm(once)


def test_an_include_line_from_a_file_included_in_a_fence_is_not_expanded_by_a_second_pass():
    """DESIGN.md: "An include line inside an included file is not expanded (one level, as the build does); it is
    escaped like any text line that reads as block syntax, so a second pass doesn't expand it either." Inside a
    fence it is left as code (tests/test_normalize.py pins that), so the second pass sees the page's own
    '{!../../docs_src/t/item.html!}' inside a fence and expands it: '<p>{{ item }}</p>' appears. Either the
    normalizer should refuse such a file (as it refuses one with a fence-closing line), or DESIGN.md must name the
    exception; as is, normalize(normalize(page)) != normalize(page)."""
    page = "```\n{!../../docs_src/q/nested.md!}\n```"
    # Fixed by refusing such a file, the first option above.
    with pytest.raises(NormalizeError, match="include line inside a fence"):
        norm(page)


def test_a_script_element_whose_attribute_holds_a_less_than_sign_is_dropped():
    """DESIGN.md: a `<script>` element is dropped up to its end tag (CommonMark HTML block 1 starts at '<script'
    followed by whitespace or '>', whatever the attributes hold). The new _RAW_ELEMENT forbids '<' in the
    attributes, so '<script data-tpl="<b>">' is not recognised, nor is it a tag: the script's source becomes page
    text ('render_secret()' below). 0e87c41's '[^>]*' dropped it."""
    page = '<script type="text/x-template" data-tpl="<b>">render_secret()</script>\n\nAfter.'
    assert "render_secret" not in norm(page)


def test_a_code_span_may_continue_from_the_page_into_an_included_file():
    """DESIGN.md rule 4: code spans are found over the whole paragraph "(included lines and the page's own form one
    paragraph)". That holds for an included file's lines followed by the page's, not for a page line followed by
    an include: 'Use `<i>a' + include of 'b</i>` c' is, in the build, one paragraph with the code span
    '`<i>a\\nb</i>`', so its tags are code and stay. Today the page line ends its paragraph at the include line, both
    backticks are read as literal, and both tags are removed: 'Use `a\\nb` c'."""
    assert norm("Use `<i>a\n{!../../docs_src/tail.md!}") == "Use `<i>a\nb</i>` c"
