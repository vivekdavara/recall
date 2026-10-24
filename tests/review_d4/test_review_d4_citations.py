"""Review of day 4 (713a52a): find_quote's word-boundary rule (DESIGN.md "Answers", "Citations")."""

from __future__ import annotations

import json

import pytest

from recall.citations import check, find_quote
from recall.retrieval import Hit


def hit(content: str) -> Hit:
    return Hit(1, "p.md", "P", "https://x/p/", (), 0, len(content), content, 1.0, 1, 1)


@pytest.mark.parametrize(
    ("quote", "text"),
    [
        (" secure by default", "The app is insecure by default."),
        ("\nsecure by default", "The app is insecure by default."),
        ("secure by default ", "Not secure by defaults."),
        ("secure by default\n", "Not secure by defaults."),
    ],
)
def test_whitespace_at_a_quotes_ends_does_not_lift_the_word_boundary(quote: str, text: str):
    """The boundary is chosen from quote[0] / quote[-1]; when that is whitespace no boundary is asked for, but the
    whitespace-flexible pattern (r"\\s+".join(quote.split())) drops the whitespace, so the match starts or ends
    inside a word: " secure by default" is accepted in "insecure by default", the day-3 review's own example.
    Models do copy a line with its newline. Correct: the boundary follows the quote's first and last non-space
    characters, so these are not found."""
    assert find_quote(quote, text) is None


def test_check_drops_a_citation_whose_quote_starts_inside_a_word_after_a_newline():
    """Through citations.check: the citation counts and its stored quote is "secure by default", cut out of
    "insecure". Correct: the claim is dropped (no citation that checks out)."""
    response = json.dumps(
        {
            "answerable": True,
            "claims": [{"text": "It is secure.", "citations": [{"passage": 1, "quote": "\nsecure by default"}]}],
        }
    )
    checked = check(response, [hit("The app is insecure by default.")])
    assert checked.claims == ()
