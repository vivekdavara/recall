"""Review findings for recall.markdown. Fails on 40a7144."""

from __future__ import annotations

import pytest

from recall import markdown


@pytest.mark.parametrize("line", ["# #", "### ###", "#  #"])
def test_a_heading_of_only_a_closing_sequence_has_an_empty_title(line: str):
    """The regex's own comment: "an optional closing run of '#' (after a space) is not part of the title";
    CommonMark makes `### ###` an empty h3. Today the closing run becomes the title ("###"), so it shows up in
    the heading path of every chunk under it."""
    h = markdown.heading(line)
    assert h is not None and h.title == ""
