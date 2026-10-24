"""Faithfulness checks: hand-written claims (not model output) that should pass and fail, and answers scored."""

from __future__ import annotations

import psycopg
import pytest
from test_retrieval import load

from recall.answer import Answer
from recall.citations import Citation, Claim
from recall.embedding import HashEmbedder
from recall.evaluation import Gold, Question, gold_chunks
from recall.faithfulness import (
    MIN_SUPPORT,
    AnswerCheck,
    ClaimCheck,
    answer_all,
    check_answer,
    check_claim,
    content_words,
    exact_terms,
    score_answers,
)
from recall.rerank import OverlapReranker
from recall.retrieval import Hit


def hit(chunk_id: int, content: str) -> Hit:
    return Hit(chunk_id, "p.md", "Page", "https://x/p/", ("Page",), 0, len(content), content, 0.5, 1, 1)


PASSAGES = [
    hit(1, "Query parameters `skip` and `limit` have default values: skip=0 and limit=10."),
    hit(2, "Raise an `HTTPException` with a status_code of 404 when the item isn't found."),
    hit(3, "Declare a `response_model` to filter the data your path operation returns."),
]


def claim(text: str, *passages: int) -> Claim:
    return Claim(text, tuple(Citation(n, n, "p.md", "https://x/p/", "q", 0, 1) for n in passages))


def test_content_words_drop_stop_words_and_plural_s():
    assert content_words("The parameters have defaults, as does the class") == {"parameter", "default", "class"}


def test_exact_terms_are_code_numbers_and_identifiers():
    text = "Use `Query(max_length=50)` and response_model_exclude_unset with HTTPException, status 404, v0.157.0."
    assert exact_terms(text) == {
        "Query(max_length=50)",
        "response_model_exclude_unset",
        "HTTPException",
        "404",
        "v0.157.0",
    }
    assert exact_terms("Plain words only, Starting with a capital.") == set()


@pytest.mark.parametrize(
    "text",
    [
        "skip and limit have default values: skip=0 and limit=10.",  # the passage's own sentence
        "The default of limit is 10 and of skip is 0.",  # reworded, same facts
        "Query parameters `skip` and `limit` have default values.",
    ],
)
def test_a_claim_that_says_what_its_passage_says_is_supported(text: str):
    check = check_claim(claim(text, 1), PASSAGES)
    assert check.supported, check


def test_a_wrong_number_or_an_invented_identifier_is_a_missing_term():
    wrong = check_claim(claim("The default of limit is 100.", 1), PASSAGES)
    assert wrong.missing_terms == ("100",) and not wrong.supported
    invented = check_claim(claim("Set response_model_skip_none to drop nulls.", 3), PASSAGES)
    assert "response_model_skip_none" in invented.missing_terms and not invented.supported


def test_a_claim_mostly_of_words_its_passage_lacks_is_unsupported():
    check = check_claim(claim("Background workers retry failed jobs with exponential backoff.", 1), PASSAGES)
    assert check.support < MIN_SUPPORT and not check.supported
    assert "backoff" in check.unsupported_words


def test_only_the_cited_passages_count():
    text = "Raise an HTTPException with status_code 404."
    assert check_claim(claim(text, 2), PASSAGES).supported
    assert not check_claim(claim(text, 1), PASSAGES).supported
    assert check_claim(claim(text, 1, 2), PASSAGES).supported  # two citations: either passage's words count


def test_a_negated_claim_passes_the_lexical_check():
    # The known limit: the passage's own words saying the opposite. Faithfulness here is lexical, not meaning.
    assert check_claim(claim("skip and limit have no default values.", 1), PASSAGES).supported


def test_a_claim_without_content_words_is_vacuously_supported():
    assert check_claim(claim("It is.", 1), PASSAGES) == ClaimCheck("It is.", 1.0, (), ())


def answer(*claims: Claim, method: str = "llm") -> Answer:
    return Answer("q", method, claims, tuple(PASSAGES), None, None)  # type: ignore[arg-type]


def test_an_answer_cites_gold_when_a_citation_is_in_a_gold_chunk():
    result = answer(claim("skip=0 and limit=10.", 1), claim("The default of limit is 100.", 1))
    check = check_answer("q1", result, frozenset({1}))
    assert check.cites_gold and check.gold_in_passages
    assert [c.supported for c in check.claims] == [True, False]
    other = check_answer("q1", result, frozenset({3}))
    assert not other.cites_gold and other.gold_in_passages
    assert not check_answer("q1", result, frozenset({99})).gold_in_passages


def test_answer_scores_by_hand():
    good = ClaimCheck("a", 1.0, (), ())
    bad = ClaimCheck("b", 0.5, ("x",), ())
    checks = [
        AnswerCheck("q1", "extractive", (good, good), True, True),
        AnswerCheck("q2", "extractive", (good, bad), False, True),
        AnswerCheck("q3", "none", (), False, False),
    ]
    s = score_answers(checks)
    assert (s.answers, s.claims, dict(s.methods)) == (3, 4, {"extractive": 2, "none": 1})
    assert s.supported_claims == 0.75 and s.fully_supported == 0.5
    assert s.cites_gold == pytest.approx(1 / 3) and s.gold_in_passages == pytest.approx(2 / 3)
    assert score_answers([]).answers == 0


def test_answer_all_answers_and_checks_every_question(conn: psycopg.Connection):
    load(conn)
    questions = [
        Question(
            "q1", "Which default values do skip and limit have?", "identifier", (Gold("query.md", "skip and limit"),)
        ),
        Question("q2", "How do I send email after a response?", "paraphrase", (Gold("tasks.md", "sending email"),)),
    ]
    gold = gold_chunks(conn, questions)
    checks = answer_all(conn, questions, gold, HashEmbedder(), OverlapReranker(), None)
    assert [c.question_id for c in checks] == ["q1", "q2"]
    assert all(c.method == "extractive" and c.claims for c in checks)
    # The extractive answer is the passages' own sentences, so every claim is supported by construction.
    assert all(cl.supported for c in checks for cl in c.claims)
    assert checks[0].cites_gold and checks[0].gold_in_passages


def test_dotted_and_hyphenated_words_are_terms_and_abbreviations_arent():
    assert exact_terms("Set max-age and call app.get, e.g. for 1,500 items (i.e. most).") == {
        "max-age",
        "app.get",
        "1,500",
    }


def test_a_term_must_be_found_as_a_whole_word():
    passage = [hit(1, "Requires Python 3.10 or later.")]
    assert check_claim(claim("Requires Python 3.1 or later.", 1), passage).missing_terms == ("3.1",)


def test_a_short_word_ending_in_s_keeps_its_s():
    assert content_words("yes") == {"yes"} and content_words("defaults") == {"default"}


@pytest.mark.parametrize(
    ("text", "supported"),
    [
        ("alpha beta gamma delta", True),  # 3 of 4 content words in the passage: 0.75, the threshold
        ("alpha beta gamma delta epsilon", False),  # 3 of 5: 0.6
    ],
)
def test_support_is_the_share_of_content_words_against_the_threshold(text: str, supported: bool):
    check = check_claim(claim(text, 1), [hit(1, "alpha beta gamma")])
    assert check.supported is supported
