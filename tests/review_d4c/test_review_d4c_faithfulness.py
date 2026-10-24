"""Review of e82579a (the day-4 second review's fix): find_term's whole-number guards and the number token
(DESIGN.md "Evaluation", "Faithfulness": a claim that copies or faithfully restates its passage is supported; one that
changes a number or identifier isn't)."""

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
    ("claimed", "passage"),
    [
        # tutorial/security/first-steps.md, as normalized
        (
            "`OAuth2PasswordBearer` inherits from `OAuth2`, which inherits from `SecurityBase`.",
            "FastAPI will know that it can use the class `OAuth2PasswordBearer` (declared in a dependency) to define "
            "the security scheme in OpenAPI because it inherits from `fastapi.security.oauth2.OAuth2`, which in turn "
            "inherits from `fastapi.security.base.SecurityBase`.",
        ),
        # advanced/stream-data.md (docs_src/stream_data/tutorial002_py310.py), as normalized
        (
            "`binary_image` is `b64decode(image_base64)`.",
            "binary_image = base64.b64decode(image_base64)",
        ),
        (
            "Old models subclass `BaseModel` from `pydantic.v1`.",
            "Old models subclass `pydantic.v1.BaseModel`.",
        ),
    ],
)
def test_an_identifier_after_a_module_name_ending_in_a_digit_is_found(claimed: str, passage: str):
    """find_term's guard (?<!\\d[.,]) is applied whenever the term's first character is a word character, letters
    included, so it refuses any term that follows "<digit>." in the passage: "OAuth2" in "...oauth2.OAuth2",
    "b64decode" in "base64.b64decode", "BaseModel" in "pydantic.v1.BaseModel". Those dots are attribute access after
    a module whose name ends in a digit, not a decimal point; the first two are in the corpus. find_quote, which this
    replaced, found them, so a claim naming a class or function without its module path is now unsupported (a
    regression). The guard exists for numbers ("500" in "1,500"). Correct: the claim is supported; the "<digit>[.,]"
    guards apply only at a term edge that is a digit (a term starting/ending with a digit)."""
    result = check_claim(claim(claimed), [hit(passage)])
    assert result.supported, result


@pytest.mark.parametrize(
    ("claimed", "passage"),
    [
        ("The lowest accepted value is 5.", "The lowest accepted value is -5."),
        ("The lowest accepted value is -5.", "The lowest accepted value is 5."),
    ],
)
def test_a_number_is_not_found_without_its_minus_sign(claimed: str, passage: str):
    """The numbers must match whole, but a minus sign is in neither: _TOKEN starts at a word character, so the
    claim's "-5" becomes the term "5", and find_term's guards don't look at "-", so "5" is found in the passage's
    "-5". A claim that flips the sign of its passage's number is supported in both directions. (Not new in e82579a,
    but the same gap it closed for "," and ".": a number found inside a different number.) Correct: unsupported
    (a "-" that isn't preceded by a word character belongs to the number, as in "-5" but not in the range "1-10")."""
    result = check_claim(claim(claimed), [hit(passage)])
    assert not result.supported, result
