"""Review 2 findings for recall.markdown. Each test fails on commit 0e87c41."""

from __future__ import annotations

import pytest

from recall import markdown


@pytest.mark.parametrize(("line", "title"), [("### ## #", "##"), ("# # #", "#"), ("## ### ##", "###")])
def test_a_title_of_hashes_before_a_closing_sequence_is_kept(line: str, title: str):
    """CommonMark: the closing sequence is the last run of '#' after a space; what is before it is the title
    (commonmark.js strips /[ \\t]+#+[ \\t]*$/ and only then an all-'#' remainder). '### ## #' is an h3 titled '##'.
    Only a remainder that is a single run of '#' ('### ###') is a bare closing sequence. Today any title made only
    of '#' characters becomes empty, so these headings lose their titles from the heading path."""
    h = markdown.heading(line)
    assert h is not None and h.title == title
