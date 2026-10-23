"""The answer pipeline: the prompt, when the model's answer is used, every fallback, and ask() end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_retrieval import DOCS, load

from recall.answer import SYSTEM_PROMPT, Answer, answer, ask, build_request, load_documents
from recall.citations import ANSWER_SCHEMA
from recall.embedding import HashEmbedder
from recall.llm import HAND_WRITTEN, Completion, ProviderError, RecordedProvider, Request, write_recording
from recall.rerank import OverlapReranker
from recall.retrieval import Hit


def hit(chunk_id: int, content: str, path: str = "p.md", heading: tuple[str, ...] = ("Page", "Part")) -> Hit:
    return Hit(chunk_id, path, "Page", f"https://x/{path}", heading, 10, 10 + len(content), content, 0.5, 1, 1)


PASSAGES = [
    hit(1, "Query parameters can have default values. Set them in the function signature."),
    hit(2, "Background tasks run after the response is sent to the client.", path="tasks.md", heading=()),
]
QUESTION = "Can query parameters have default values?"


class Replies:
    """A provider that returns one fixed text (or raises), and remembers the requests."""

    name = "replies"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text, self.error = text, error
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Completion:
        self.requests.append(request)
        if self.error:
            raise self.error
        return Completion(self.text, "model-x")


def reply(*claims: tuple[str, int, str], answerable: bool = True) -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "claims": [{"text": t, "citations": [{"passage": p, "quote": q}]} for t, p, q in claims],
        }
    )


def test_the_prompt_numbers_the_passages_with_their_page_and_headings():
    request = build_request(QUESTION, PASSAGES)
    assert request.system == SYSTEM_PROMPT and request.schema == ANSWER_SCHEMA
    assert request.prompt == (
        "Passages:\n\n"
        "[1] p.md (Page > Part)\nQuery parameters can have default values. Set them in the function signature.\n\n"
        "[2] tasks.md\nBackground tasks run after the response is sent to the client.\n\n"
        f"Question: {QUESTION}"
    )
    assert build_request(QUESTION, PASSAGES).key() == request.key()  # deterministic, so recordings are found again


def test_the_models_checked_claims_are_the_answer():
    provider = Replies(reply(("Yes, they can.", 1, "can have default values")))
    result = answer(QUESTION, PASSAGES, provider, OverlapReranker())
    assert (result.method, result.model, result.provider, result.problems) == ("llm", "model-x", "replies", ())
    assert result.text == "Yes, they can. [1]"
    assert provider.requests == [build_request(QUESTION, PASSAGES)]


def test_claims_that_fail_the_check_are_dropped_and_reported():
    provider = Replies(reply(("Yes.", 1, "can have default values"), ("Invented.", 2, "not in the passage at all")))
    result = answer(QUESTION, PASSAGES, provider, OverlapReranker())
    assert result.method == "llm" and [c.text for c in result.claims] == ["Yes."]
    assert result.problems == (
        "claim 2, citation 1 dropped: the quote isn't in passage 2",
        "claim 2 dropped: no citation that checks out",
    )


@pytest.mark.parametrize(
    ("provider", "problem"),
    [
        (Replies(error=ProviderError("the provider is down")), "the provider is down"),
        (Replies("not json"), "the answer isn't JSON"),
        (Replies(reply(("Invented.", 1, "words that are not there"))), "no claim in the model's answer had a citation"),
        (Replies(reply()), "no claim in the model's answer had a citation"),
    ],
)
def test_without_a_usable_model_answer_the_answer_is_extractive(provider: Replies, problem: str):
    result = answer(QUESTION, PASSAGES, provider, OverlapReranker())
    assert result.method == "extractive" and result.model is None and result.provider == "replies"
    assert any(problem in p for p in result.problems), result.problems
    assert result.claims[0].text == "Query parameters can have default values."


def test_no_provider_is_the_extractive_answer_with_no_problems():
    result = answer(QUESTION, PASSAGES, None, OverlapReranker())
    assert (result.method, result.provider, result.problems) == ("extractive", None, ())
    assert result.text.startswith("Query parameters can have default values. [1]")


def test_a_missing_recording_falls_back_to_the_extractive_answer(tmp_path: Path):
    result = answer(QUESTION, PASSAGES, RecordedProvider(tmp_path), OverlapReranker())
    assert result.method == "extractive"
    assert result.problems == (f"no recording for this question's prompt in {tmp_path}",)


def test_a_recording_of_the_prompt_is_replayed(tmp_path: Path):
    text = reply(("Yes, set them in the signature.", 1, "Set them in the function signature."))
    write_recording(tmp_path, build_request(QUESTION, PASSAGES), Completion(text, HAND_WRITTEN), "test")
    result = answer(QUESTION, PASSAGES, RecordedProvider(tmp_path), OverlapReranker())
    assert (result.method, result.model) == ("llm", HAND_WRITTEN)
    assert result.text == "Yes, set them in the signature. [1]"


def test_when_the_model_says_the_passages_dont_answer_there_is_no_answer():
    result = answer("What is the capital of France?", PASSAGES, Replies(reply(answerable=False)), OverlapReranker())
    assert (result.method, result.claims, result.model) == ("none", (), "model-x")


def test_no_passages_no_answer_and_no_model_call():
    provider = Replies(reply())
    assert answer(QUESTION, [], provider, OverlapReranker()) == Answer(QUESTION, "none", (), (), "replies", None)
    assert provider.requests == []


def test_passages_without_sentences_give_no_answer():
    result = answer(QUESTION, [hit(1, "```\ncode()\n```")], None, OverlapReranker())
    assert (result.method, result.claims) == ("none", ())


def test_text_lists_each_passage_once_per_claim():
    two = json.dumps(
        {
            "answerable": True,
            "claims": [
                {
                    "text": "Both.",
                    "citations": [
                        {"passage": 2, "quote": "Background tasks run"},
                        {"passage": 1, "quote": "default values"},
                        {"passage": 1, "quote": "function signature"},
                    ],
                }
            ],
        }
    )
    assert answer(QUESTION, PASSAGES, Replies(two), OverlapReranker()).text == "Both. [2] [1]"


# ask(): retrieval, the reranker and the answer, on a small corpus in the test database.


def test_ask_answers_from_the_reranked_top_passages(conn: psycopg.Connection):
    load(conn)
    result, timings = ask(
        conn, "Which default values do skip and limit have?", HashEmbedder(), OverlapReranker(), None, passages=2
    )
    assert result.method == "extractive" and len(result.passages) == 2
    assert result.passages[0].path == "query.md"
    assert result.claims[0].text == "skip and limit have default values."
    citation = result.claims[0].citations[0]
    page = next(d for d in DOCS if d.path == citation.path)
    assert page.content[citation.start : citation.end] == citation.quote
    assert timings.retrieve > 0 and timings.rerank > 0 and timings.answer > 0
    assert timings.total == pytest.approx(timings.retrieve + timings.rerank + timings.answer)


def test_ask_without_the_reranker_answers_from_the_retrieval_order(conn: psycopg.Connection):
    load(conn)

    class Reversed:
        name = "reversed"

        def score(self, question: str, texts: Any) -> list[float]:
            return [float(i) for i in range(len(texts))]  # prefers the last

    question = "Which default values do skip and limit have?"
    plain, timings = ask(conn, question, HashEmbedder(), Reversed(), None, passages=3, use_reranker=False)
    reranked, _ = ask(conn, question, HashEmbedder(), Reversed(), None, passages=3, depth=3)
    assert timings.rerank == 0.0
    assert [h.chunk_id for h in reranked.passages] == [h.chunk_id for h in reversed(plain.passages)]


def test_ask_checks_its_sizes(conn: psycopg.Connection):
    with pytest.raises(ValueError, match="passages must be from 1 to depth"):
        ask(conn, "q", HashEmbedder(), OverlapReranker(), None, passages=6, depth=5)
    with pytest.raises(ValueError, match="passages must be from 1 to depth"):
        ask(conn, "q", HashEmbedder(), OverlapReranker(), None, passages=0)


def test_load_documents_reads_the_pages_asked_for(conn: psycopg.Connection):
    load(conn)
    assert load_documents(conn, ["query.md", "query.md", "missing.md"]) == {"query.md": DOCS[0].content}
    assert load_documents(conn, []) == {}
