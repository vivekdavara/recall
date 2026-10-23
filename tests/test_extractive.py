"""The extractive answer: which text is a sentence, which sentences are chosen, and that each cites its own span."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recall import markdown
from recall.citations import ANSWER_SCHEMA, check
from recall.extractive import candidates, extract, sentences
from recall.rerank import OverlapReranker
from recall.retrieval import Hit


def texts(content: str) -> list[str]:
    return [content[s:e] for s, e in sentences(content)]


def test_a_paragraph_splits_into_sentences():
    assert texts("One sentence here. Another one! And a third?  \nStill the third's line.") == [
        "One sentence here.",
        "Another one!",
        "And a third?",
        "Still the third's line.",
    ]


def test_a_sentence_may_end_inside_quotes_code_or_emphasis():
    assert texts('It is "done." Then `x.` And **bold.** Last') == [
        'It is "done."',
        "Then `x.`",
        "And **bold.**",
        "Last",
    ]


def test_abbreviations_dont_end_a_sentence():
    assert texts("Use a type, e.g. int or str. Then i.e. this. Done.") == [
        "Use a type, e.g. int or str.",
        "Then i.e. this.",
        "Done.",
    ]


def test_fenced_code_and_headings_are_not_sentences():
    content = "Intro text.\n\n## Section\n\n```python\nx = 1. Not prose.\n```\n\nAfter the code."
    assert texts(content) == ["Intro text.", "After the code."]


def test_a_list_item_is_a_block_of_its_own_without_its_marker():
    content = "Includes:\n\n* Email notifications\n    continued here.\n* Processing data\n1. First step"
    assert texts(content) == ["Includes:", "Email notifications\n    continued here.", "Processing data", "First step"]


def test_blank_lines_separate_blocks():
    assert texts("no period here\n\nnext block") == ["no period here", "next block"]


def test_empty_and_code_only_chunks_have_no_sentences():
    assert sentences("") == []
    assert sentences("```\ncode. More code.\n```") == []
    assert sentences("   \n\n  ") == []


def hit(chunk_id: int, content: str, start: int = 100) -> Hit:
    return Hit(
        chunk_id,
        f"p{chunk_id}.md",
        "Page",
        f"https://x/p{chunk_id}/",
        ("Page",),
        start,
        start + len(content),
        content,
        0.5,
        1,
        1,
    )


PASSAGES = [
    hit(1, "Background tasks run after the response is sent. They are useful for email.\n\n```\ncode()\n```"),
    hit(2, "Tip:\n\nQuery parameters have default values. You can make query parameters optional too."),
]


def test_candidates_are_the_sentences_of_at_least_four_words():
    assert [(c.passage, c.text) for c in candidates(PASSAGES)] == [
        (1, "Background tasks run after the response is sent."),
        (1, "They are useful for email."),
        (2, "Query parameters have default values."),
        (2, "You can make query parameters optional too."),
    ]


def test_extract_takes_the_best_sentences_best_first_each_citing_its_span():
    claims = extract("query parameters default values", PASSAGES, OverlapReranker(), n=2)
    assert [c.text for c in claims] == [
        "Query parameters have default values.",
        "You can make query parameters optional too.",
    ]
    for claim in claims:
        (citation,) = claim.citations
        passage = PASSAGES[citation.passage - 1]
        assert passage.content[citation.start - passage.start : citation.end - passage.start] == citation.quote
        assert (citation.chunk_id, citation.path, citation.url) == (passage.chunk_id, passage.path, passage.url)


def test_equal_scores_keep_passage_order():
    claims = extract("nothing in common", PASSAGES, OverlapReranker(), n=4)
    assert [c.citations[0].passage for c in claims] == [1, 1, 2, 2]


def test_no_sentences_no_claims():
    assert extract("q", [hit(1, "```\ncode\n```")], OverlapReranker()) == []
    assert extract("q", [], OverlapReranker()) == []


class Wrong:
    name = "wrong"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        return self.scores


@pytest.mark.parametrize("scores", [[1.0], [1.0, 2.0, 3.0, float("nan")]])
def test_a_reranker_that_gives_the_wrong_scores_is_an_error(scores: list[float]):
    with pytest.raises(ValueError, match="gave"):
        extract("q", PASSAGES, Wrong(scores))


def test_the_extractive_answer_passes_the_citation_check():
    import json

    claims = extract("query parameters default values", PASSAGES, OverlapReranker())
    response = json.dumps(
        {
            "answerable": True,
            "claims": [
                {"text": c.text, "citations": [{"passage": x.passage, "quote": x.quote} for x in c.citations]}
                for c in claims
            ],
        }
    )
    assert set(ANSWER_SCHEMA["required"]) == {"answerable", "claims"}
    checked = check(response, PASSAGES)
    assert checked.problems == () and checked.claims == tuple(claims)


# Any text: every span is a trimmed, non-empty slice outside fenced code and headings.
lines = st.sampled_from(
    [
        "",
        "  ",
        "Some prose. More prose!",
        "* item one.",
        "1. step",
        "## Heading",
        "```",
        "~~~",
        "code. inside?",
        "e.g. this",
        "end.",
        "   indented text",
    ]
) | st.text(max_size=30)


@settings(max_examples=400, deadline=None)
@given(st.lists(lines, max_size=25))
def test_every_sentence_is_a_trimmed_slice_of_prose(parts: list[str]):
    content = "\n".join(parts)
    fenced_or_heading = set()
    for line in markdown.scan(content):
        text = content[line.start : line.end]
        if line.fence is not None or markdown.heading(text) is not None:
            fenced_or_heading.update(range(line.start, line.end + 1))
    previous_end = 0
    for s, e in sentences(content):
        assert previous_end <= s < e <= len(content)
        text = content[s:e]
        assert text == text.strip() and text
        assert not fenced_or_heading.intersection(range(s, e))
        previous_end = e


# A code block too long for one chunk is split across chunks: the second chunk starts inside it. Read alone, its code
# is prose and its closing fence opens a "block" that hides the prose after it; read in its document, neither.
SPLIT_DOCUMENT = (
    "Declare the model first. It holds the fields.\n\n```python\nclass Item(BaseModel):\n    name: str\n"
    "    price: float = 1.0. Not a sentence.\n```\n\nThen use it as a parameter of the function."
)
CUT = SPLIT_DOCUMENT.index("    price")


def test_a_chunk_that_starts_inside_a_code_block_is_read_in_its_document():
    second = hit(2, SPLIT_DOCUMENT[CUT:], start=CUT)
    # Alone, the code line is a "sentence" and the prose after the closing fence is lost.
    assert [c.text for c in candidates([second])] == ["price: float = 1.0."]
    in_document = [c.text for c in candidates([second], {second.path: SPLIT_DOCUMENT})]
    assert in_document == ["Then use it as a parameter of the function."]
    first = hit(1, SPLIT_DOCUMENT[: CUT - 1], start=0)
    assert [c.text for c in candidates([first], {first.path: SPLIT_DOCUMENT})] == [
        "Declare the model first.",
        "It holds the fields.",
    ]


def test_a_page_that_no_longer_holds_the_passage_is_not_used():
    second = hit(2, SPLIT_DOCUMENT[CUT:], start=CUT)
    stale = {second.path: "something else entirely " * 20}
    assert candidates([second], stale) == candidates([second])


def test_a_span_cut_mid_line_has_no_list_marker_and_no_heading_check_on_its_part():
    document = "* first item is long enough. Second sentence here."
    at = document.index("Second")
    assert [document[s:e] for s, e in sentences(document, at, len(document))] == ["Second sentence here."]
    assert [document[s:e] for s, e in sentences(document, 2, len(document))] == [
        "first item is long enough.",
        "Second sentence here.",
    ]


@settings(max_examples=300, deadline=None)
@given(st.lists(lines, max_size=25), st.data())
def test_sentences_of_a_span_are_the_documents_sentences_inside_it(parts: list[str], data: st.DataObject):
    document = "\n".join(parts)
    start = data.draw(st.integers(0, len(document)))
    end = data.draw(st.integers(start, len(document)))
    inside = sentences(document, start, end)
    for s, e in inside:
        assert start <= s < e <= end
    # A span cut at blank lines (where chunks of whole blocks are cut) gives exactly the document's sentences in it.
    before = document[:start]
    after = document[end:]
    at_block_edges = (start == 0 or before.endswith("\n\n")) and (end == len(document) or after.startswith("\n\n"))
    if at_block_edges:
        assert inside == [(s, e) for s, e in sentences(document) if start <= s and e <= end]
