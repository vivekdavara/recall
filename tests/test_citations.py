"""The citation check: what counts as a citation, the spans it gives, and that no response can make it raise."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recall.citations import MIN_QUOTE_CHARS, Citation, Claim, MalformedAnswer, check, find_quote
from recall.retrieval import Hit

DOCUMENT = (
    "# Query Parameters\n\nWhen you declare other function parameters that are not part of the path parameters,"
    ' they are automatically interpreted as "query" parameters.\n\n## Defaults\n\nAs query parameters are not a'
    " fixed part of a path, they can be optional\nand can have default values.\n"
)


def hit(chunk_id: int, start: int, end: int, path: str = "tutorial/query-params.md") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        path=path,
        title="Query Parameters",
        url=f"https://fastapi.tiangolo.com/{path.removesuffix('.md')}/",
        heading_path=("Query Parameters",),
        start=start,
        end=end,
        content=DOCUMENT[start:end],
        score=1.0,
        full_text_rank=1,
        vector_rank=1,
    )


SPLIT = DOCUMENT.index("## Defaults")
PASSAGES = [hit(10, 0, SPLIT), hit(11, SPLIT, len(DOCUMENT))]


def answer(*claims: dict[str, Any], answerable: bool = True) -> str:
    return json.dumps({"answerable": answerable, "claims": list(claims)})


def claim(text: str, *citations: tuple[Any, Any]) -> dict[str, Any]:
    return {"text": text, "citations": [{"passage": p, "quote": q} for p, q in citations]}


def test_a_quote_from_the_cited_passage_gives_the_documents_span():
    checked = check(answer(claim("Query parameters can be optional.", (2, "they can be optional"))), PASSAGES)
    assert checked.problems == ()
    ((c,),) = [cl.citations for cl in checked.claims]
    assert c.passage == 2 and c.chunk_id == 11 and c.path == "tutorial/query-params.md"
    assert DOCUMENT[c.start : c.end] == c.quote == "they can be optional"


def test_a_rewrapped_quote_counts_and_the_span_is_the_passages_own_text():
    checked = check(answer(claim("Optional.", (2, "they can be optional and can have   default values"))), PASSAGES)
    (c,) = checked.claims[0].citations
    assert c.quote == "they can be optional\nand can have default values"
    assert DOCUMENT[c.start : c.end] == c.quote


@pytest.mark.parametrize(
    ("citation", "reason"),
    [
        ((3, "they can be optional"), "passage 3 isn't one of the 2 given"),
        ((0, "they can be optional"), "passage 0 isn't one of the 2 given"),
        ((True, "they can be optional"), "passage True isn't one of the 2 given"),
        ((2.0, "they can be optional"), "passage 2.0 isn't one of the 2 given"),
        (("2", "they can be optional"), "passage '2' isn't one of the 2 given"),
        ((2, "Optional values"), "the quote isn't in passage 2"),  # case matters
        ((1, "they can be optional"), "the quote isn't in passage 1"),  # in the document, but not in passage 1
        ((2, "they may be optional"), "the quote isn't in passage 2"),
        ((2, "path,  "), f"at least {MIN_QUOTE_CHARS} characters"),
        ((2, None), f"at least {MIN_QUOTE_CHARS} characters"),
        ((2, "x" * 2001), "longer than 2000"),
    ],
)
def test_a_citation_that_doesnt_check_out_is_dropped_with_its_reason(citation: tuple[Any, Any], reason: str):
    checked = check(answer(claim("A claim.", citation), claim("Kept.", (2, "default values"))), PASSAGES)
    assert [cl.text for cl in checked.claims] == ["Kept."]
    assert any(reason in p for p in checked.problems), checked.problems
    assert "claim 1 dropped: no citation that checks out" in checked.problems


def test_a_claim_keeps_the_citations_that_check_out():
    checked = check(
        answer(claim("Both.", (2, "default values"), (1, "nope nope nope"), (1, "query parameters"))), PASSAGES
    )
    assert [c.passage for c in checked.claims[0].citations] == [2]
    assert checked.problems == (
        "claim 1, citation 2 dropped: the quote isn't in passage 1",
        "claim 1, citation 3 dropped: the quote isn't in passage 1",
    )


def test_the_same_citation_twice_is_kept_once():
    checked = check(answer(claim("Twice.", (2, "default values"), (2, "default values"))), PASSAGES)
    assert len(checked.claims[0].citations) == 1


def test_an_unanswerable_answer_has_no_claims():
    checked = check(answer(claim("Anyway.", (2, "default values")), answerable=False), PASSAGES)
    assert (checked.answerable, checked.claims) == (False, ())
    assert checked.problems == ("the answer says the passages don't answer the question, so its claims were ignored",)
    assert check(answer(answerable=False), PASSAGES) == check(answer(answerable=False), PASSAGES)
    assert check(answer(answerable=False), PASSAGES).problems == ()


@pytest.mark.parametrize(
    "response",
    [
        "",
        "not json",
        "[]",
        "null",
        '{"claims": []}',
        '{"answerable": 1, "claims": []}',
        '{"answerable": true}',
        '{"answerable": true, "claims": {}}',
        "[" * 100_000,
    ],
)
def test_a_response_that_isnt_an_answer_is_malformed(response: str):
    with pytest.raises(MalformedAnswer):
        check(response, PASSAGES)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("text", "not an object with text"),
        ({"text": "x"}, "not an object with text"),
        ({"text": 5, "citations": []}, "not an object with text"),
        ({"text": "   ", "citations": [{"passage": 2, "quote": "default values"}]}, "no text"),
        ({"text": "x", "citations": ["2"]}, "not an object"),
    ],
)
def test_a_malformed_claim_is_dropped_not_fatal(raw: Any, reason: str):
    checked = check(json.dumps({"answerable": True, "claims": [raw, claim("Kept.", (2, "default values"))]}), PASSAGES)
    assert [c.text for c in checked.claims] == ["Kept."]
    assert any(reason in p for p in checked.problems), checked.problems


def test_claim_text_is_one_line():
    checked = check(answer(claim("  Two\n lines. ", (2, "default values"))), PASSAGES)
    assert checked.claims == (Claim("Two lines.", checked.claims[0].citations),)


def test_only_the_first_claims_and_citations_are_checked():
    many = [claim(f"Claim {i}.", (2, "default values")) for i in range(25)]
    checked = check(answer(*many), PASSAGES)
    assert len(checked.claims) == 20
    assert checked.problems[0] == "only the first 20 of 25 claims were checked"
    six = claim("Six.", *[(1, "nope nope nope")] * 5, (2, "default values"))
    assert check(answer(six), PASSAGES).claims == ()  # the sixth citation is past the limit of 5


def test_find_quote_prefers_the_exact_text_and_the_first_occurrence():
    assert find_quote("ab cd", "xx ab  cd ab cd") == (10, 15)
    assert find_quote("ab cd", "xx ab \n cd") == (3, 10)
    assert find_quote("   ", "anything") is None
    assert find_quote("a.c", "abc") is None  # regex characters are literal


# Hypothesis: any response at all is checked without raising anything but MalformedAnswer, and every citation that
# comes back is a span of its own passage.

json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-3, 5) | st.floats(allow_nan=False) | st.text(max_size=40),
    lambda inner: (
        st.lists(inner, max_size=4)
        | st.dictionaries(
            st.sampled_from(["answerable", "claims", "text", "citations", "passage", "quote", "x"]), inner, max_size=4
        )
    ),
    max_leaves=30,
)


@settings(max_examples=300, deadline=None)
@given(json_values)
def test_no_response_makes_the_check_raise(value: Any):
    try:
        checked = check(json.dumps(value), PASSAGES)
    except MalformedAnswer:
        return
    for c in checked.claims:
        for citation in c.citations:
            assert_is_span(citation)


def assert_is_span(c: Citation) -> None:
    passage = PASSAGES[c.passage - 1]
    assert passage.start <= c.start < c.end <= passage.end
    assert DOCUMENT[c.start : c.end] == c.quote


@settings(max_examples=300, deadline=None)
@given(st.data())
def test_every_quote_taken_from_a_passage_checks_out(data: st.DataObject):
    number = data.draw(st.integers(1, len(PASSAGES)))
    text = PASSAGES[number - 1].content
    start = data.draw(st.integers(0, len(text) - 1))
    end = data.draw(st.integers(start + 1, len(text)))
    quote = text[start:end]
    # Re-wrapped as a model might: every run of whitespace replaced by one space.
    if data.draw(st.booleans()):
        quote = " ".join(quote.split())
    checked = check(answer(claim("Taken.", (number, quote))), PASSAGES)
    if len("".join(quote.split())) < MIN_QUOTE_CHARS:
        assert checked.claims == ()
        return
    (c,) = checked.claims[0].citations
    assert_is_span(c)
    assert " ".join(c.quote.split()) == " ".join(quote.split())
