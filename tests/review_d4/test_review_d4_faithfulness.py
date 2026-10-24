"""Review of day 4 (713a52a): the faithfulness check's exact terms (DESIGN.md "Evaluation", "Faithfulness")."""

from __future__ import annotations

from recall.citations import Citation, Claim
from recall.faithfulness import check_claim, exact_terms
from recall.retrieval import Hit


def hit(content: str) -> Hit:
    return Hit(1, "p.md", "P", "https://x/p/", (), 0, len(content), content, 1.0, 1, 1)


def claim(text: str) -> Claim:
    return Claim(text, (Citation(1, 1, "p.md", "https://x/p/", "x", 0, 1),))


def test_a_number_with_a_thousands_separator_must_be_found_as_written():
    """_TOKEN is \\w+(?:[.\\-]\\w+)*, so "1,500" is two terms, "1" and "500", each looked for on its own: a claim that
    changes "1,000" to "1,500" passes against a passage that holds "1,000" and, elsewhere, "500". DESIGN: every exact
    term (numbers among them) "must be found as written", which is what "catches a claim that changes a default".
    Correct: "1,500" is one term, not in the passage, so the claim is unsupported."""
    passage = hit("By default, up to 1,000 items are returned. Each page holds 500 of them.")
    result = check_claim(claim("By default, up to 1,500 items are returned."), [passage])
    assert not result.supported, result


def test_latin_abbreviations_are_not_exact_terms():
    """ "e.g." and "i.e." have a dot inside, so they become exact terms the cited passage must hold: a claim that
    writes "e.g." where the passage wrote "for example" is unsupported however faithful it is, and models write
    "e.g." all the time. They are not names, numbers or identifiers (the docstring's "dotted names"). Correct: no
    exact term from them."""
    assert exact_terms("Declare it with a default, e.g. None, i.e. optional.") == set()
