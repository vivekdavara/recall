"""Faithfulness checks on answers: does each claim say only what the passages it cites say, and does the answer cite
the passages that hold the answer?

DESIGN.md "Evaluation" is the specification. The citation check (recall.citations) already proves each quote is in its
passage. These checks go a step further and stay lexical, so they need no model: most of a claim's content words must
be in the passages it cites, and every number and identifier in it must be there exactly. They catch a claim that
adds what its passages don't hold (a wrong default, an invented parameter); they can't catch one that uses the
passages' own words to say something else ("q is not required" from a passage saying "q is required").
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import psycopg

from recall.answer import Answer, ask
from recall.citations import Claim, find_quote
from recall.embedding import Embedder
from recall.evaluation import Question
from recall.llm import Provider
from recall.rerank import Reranker
from recall.retrieval import Hit

# A claim is supported if at least this share of its distinct content words is in the passages it cites, and every
# exact term is. Not tuned on model answers (there are no recorded ones yet); the tests show what passes and fails.
MIN_SUPPORT = 0.75

# Common English words that carry no claim of their own.
STOP_WORDS = frozenset(
    """a about above after again all also an and any are as at be because been before being below between both but by
    can could did do does doing down during each few for from further had has have having here how i if in into is it
    its itself just me more most my no nor not now of off on once only or other our out over own same should so some
    such than that the their them then there these they this those through to too under until up use used uses using
    very was we were what when where which while who why will with would you your""".split()  # noqa: SIM905 (a word list)
)

_WORD = re.compile(r"\w+")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_TOKEN = re.compile(r"\w+(?:[.\-]\w+)*")


def _stem(word: str) -> str:
    # Enough to match "parameters" with "parameter" and "defaults" with "default"; not "class" or "status".
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")) else word


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in STOP_WORDS}


def exact_terms(text: str) -> set[str]:
    """What a claim must not change: code spans (`...`), and words with a digit, an underscore, a dot or dash inside,
    or a capital letter after the first (numbers, versions, snake_case, dotted names, CamelCase, HTTPException)."""
    terms = {m.group(1).strip() for m in _CODE_SPAN.finditer(text) if m.group(1).strip()}
    outside = _CODE_SPAN.sub(" ", text)
    for token in _TOKEN.findall(outside):
        if (
            any(c.isdigit() for c in token)
            or "_" in token
            or "." in token
            or "-" in token
            or any(c.isupper() for c in token[1:])
        ):
            terms.add(token)
    return terms


@dataclass(frozen=True)
class ClaimCheck:
    text: str
    support: float  # the share of the claim's distinct content words found in the passages it cites
    unsupported_words: tuple[str, ...]
    missing_terms: tuple[str, ...]  # exact terms none of the cited passages holds

    @property
    def supported(self) -> bool:
        return self.support >= MIN_SUPPORT and not self.missing_terms


def check_claim(claim: Claim, passages: Sequence[Hit]) -> ClaimCheck:
    """Check a claim against the passages its citations name (by their 1-based numbers in `passages`)."""
    cited = "\n\n".join(passages[n - 1].content for n in dict.fromkeys(c.passage for c in claim.citations))
    words = content_words(claim.text)
    found = content_words(cited)
    unsupported = sorted(words - found)
    support = 1.0 if not words else (len(words) - len(unsupported)) / len(words)
    missing = sorted(t for t in exact_terms(claim.text) if find_quote(t, cited) is None)
    return ClaimCheck(claim.text, support, tuple(unsupported), tuple(missing))


@dataclass(frozen=True)
class AnswerCheck:
    question_id: str
    method: str
    claims: tuple[ClaimCheck, ...]
    cites_gold: bool  # some citation is in a gold chunk
    gold_in_passages: bool  # some passage the answer was written from is a gold chunk


def check_answer(question_id: str, result: Answer, gold: frozenset[int]) -> AnswerCheck:
    claims = tuple(check_claim(c, result.passages) for c in result.claims)
    cites_gold = any(x.chunk_id in gold for c in result.claims for x in c.citations)
    gold_in_passages = any(h.chunk_id in gold for h in result.passages)
    return AnswerCheck(question_id, result.method, claims, cites_gold, gold_in_passages)


@dataclass(frozen=True)
class AnswerScores:
    answers: int
    methods: Mapping[str, int]  # how many answers each method wrote (llm, extractive, none)
    claims: int
    supported_claims: float  # the share of all claims that are supported
    fully_supported: float  # the share of answers with claims whose claims are all supported
    cites_gold: float  # the share of answers citing a gold chunk
    gold_in_passages: float  # the share of answers written from passages that include a gold chunk

    def to_json(self) -> dict[str, object]:
        return {
            "answers": self.answers,
            "methods": dict(self.methods),
            "claims": self.claims,
            "supported_claims": self.supported_claims,
            "fully_supported": self.fully_supported,
            "cites_gold": self.cites_gold,
            "gold_in_passages": self.gold_in_passages,
        }


def score_answers(checks: Sequence[AnswerCheck]) -> AnswerScores:
    n = len(checks)
    methods: dict[str, int] = {}
    for c in checks:
        methods[c.method] = methods.get(c.method, 0) + 1
    claims = [cl for c in checks for cl in c.claims]
    with_claims = [c for c in checks if c.claims]
    return AnswerScores(
        n,
        methods,
        len(claims),
        sum(cl.supported for cl in claims) / len(claims) if claims else 0.0,
        sum(all(cl.supported for cl in c.claims) for c in with_claims) / len(with_claims) if with_claims else 0.0,
        sum(c.cites_gold for c in checks) / n if n else 0.0,
        sum(c.gold_in_passages for c in checks) / n if n else 0.0,
    )


def answer_all(
    conn: psycopg.Connection,
    questions: Sequence[Question],
    gold: Mapping[str, frozenset[int]],
    embedder: Embedder | None,
    reranker: Reranker,
    provider: Provider | None,
) -> list[AnswerCheck]:
    """Answer every question as ask() does by default (hybrid, reranked, 5 passages) and check each answer."""
    checks = []
    for q in questions:
        result, _ = ask(conn, q.question, embedder, reranker, provider)
        checks.append(check_answer(q.id, result, gold[q.id]))
    return checks
