"""Review of the day-4 fixes (d1ed563): the faithfulness check's number token (DESIGN.md "Evaluation",
"Faithfulness": every exact term "must be found as written, as a whole word (... numbers, "1,500" as one ...)")."""

from __future__ import annotations

import pytest

from recall.citations import Citation, Claim
from recall.faithfulness import check_claim
from recall.retrieval import Hit


def hit(content: str) -> Hit:
    return Hit(1, "p.md", "P", "https://x/p/", (), 0, len(content), content, 1.0, 1, 1)


def claim(text: str) -> Claim:
    return Claim(text, (Citation(1, 1, "p.md", "https://x/p/", "x", 0, 1),))


@pytest.mark.parametrize(
    "text",
    [
        "The limit is 1,024MB per upload.",
        "It serves 1,000s of requests a second.",
        "Images up to 2,048px wide are accepted.",
    ],
)
def test_a_claim_copying_a_number_with_a_unit_glued_on_is_supported(text: str):
    """The new number alternative of _TOKEN, \\d{1,3}(?:,\\d{3})+, stops at the separators, so "1,024MB" becomes
    the terms "1,024" and "MB"; find_quote then wants each as a whole word, and in the passage "1,024" is followed by
    "M" and "MB" preceded by "4", so neither is found. A claim that is the passage's own sentence, word for word, is
    unsupported. Before 4742ce8 the tokens were "1" and "024MB", both found, and this passed (a regression).
    Correct: a verbatim claim is supported (the number and its unit stay one term, or are matched as they are)."""
    result = check_claim(claim(text), [hit(text)])
    assert result.supported, result


@pytest.mark.parametrize(
    ("claimed", "passage"),
    [
        ("Each page holds up to 500 items.", "Each page holds up to 1,500 items."),
        ("Up to 1,000 rows are read.", "Up to 1,000,000 rows are read."),
        ("The timeout is 10 seconds.", "The timeout is 10.5 seconds."),
    ],
)
def test_a_number_is_not_found_inside_a_bigger_number_in_the_passage(claimed: str, passage: str):
    """4742ce8 made "1,500" one term on the claim's side only. The passage side is find_quote, whose word boundary is
    (?<!\\w) / (?!\\w), and "," and "." are not word characters: "500" is "found" in "1,500", "1,000" in
    "1,000,000", "10" in "10.5". A claim that changes the passage's 1,500 to 500 is supported, the very change the
    exact terms exist to catch, and DESIGN now says numbers are found "as a whole word ... "1,500" as one".
    Correct: the claim is unsupported (missing_terms holds the number)."""
    result = check_claim(claim(claimed), [hit(passage)])
    assert not result.supported, result
