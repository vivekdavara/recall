"""Check an answer's citations against the passages it was written from.

DESIGN.md "Answers" is the specification. The model is asked for JSON: whether the passages answer the question, and
a list of claims, each citing passages by number with a quote copied from the passage. A citation counts only if its
passage is one of those retrieved and its quote is found in that passage's text; the citation then points at the
exact span of the source page (offsets into the document's normalized text). A claim with no citation that counts is
dropped. What was dropped, and why, is reported with the answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from recall.retrieval import Hit

# The JSON schema the model's answer must follow (structured output). Passage numbers are 1-based, as the prompt
# numbers the passages.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"passage": {"type": "integer"}, "quote": {"type": "string"}},
                            "required": ["passage", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answerable", "claims"],
    "additionalProperties": False,
}

# A quote must hold at least this many characters other than whitespace: one word ("the") would match nearly any
# passage and show nothing. An identifier such as response_model is long enough.
MIN_QUOTE_CHARS = 8
# Bounds on what one answer may hold, so a runaway response can't make the check slow.
MAX_CLAIMS = 20
MAX_CITATIONS = 5
MAX_QUOTE_CHARS = 2000


@dataclass(frozen=True)
class Citation:
    passage: int  # the passage's 1-based number in the prompt
    chunk_id: int
    path: str
    url: str
    quote: str  # the passage's own text that the quote matched (document[start:end])
    start: int  # offsets into the document's normalized text
    end: int


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Checked:
    answerable: bool
    claims: tuple[Claim, ...]
    problems: tuple[str, ...]  # each citation or claim dropped, and why


class MalformedAnswer(ValueError):
    """The response isn't JSON of the answer's shape at all, so nothing in it can be checked."""


def find_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` occurs in `text`: exactly, or else with any run of whitespace standing for any other.

    Models often re-wrap the text they copy (a newline becomes a space); that is still the passage's text. Anything
    else (a changed word, different punctuation) is not found. A quote that starts or ends with a word character
    must start or end a word there too: "secure by default" isn't in "insecure by default" (the day-3 review). The
    first occurrence counts.
    """
    words = quote.split()
    if not words:
        return None
    before = r"(?<!\w)" if re.match(r"\w", quote[0]) else ""
    after = r"(?!\w)" if re.match(r"\w", quote[-1]) else ""
    for body in (re.escape(quote), r"\s+".join(re.escape(w) for w in words)):
        m = re.search(before + body + after, text)
        if m:
            return m.start(), m.end()
    return None


def check(response: str, passages: Sequence[Hit]) -> Checked:
    """Parse the model's JSON and keep each claim with the citations that check out. Raises MalformedAnswer only
    when the response as a whole isn't an answer; any single bad claim or citation is dropped and reported."""
    try:
        data = json.loads(response)
    except (ValueError, RecursionError) as e:
        raise MalformedAnswer(f"the answer isn't JSON: {e}") from None
    if (
        not isinstance(data, dict)
        or type(data.get("answerable")) is not bool
        or not isinstance(data.get("claims"), list)
    ):
        raise MalformedAnswer("the answer isn't an object with answerable (true or false) and a list of claims")
    problems: list[str] = []
    raw_claims: list[Any] = data["claims"]
    if len(raw_claims) > MAX_CLAIMS:
        problems.append(f"only the first {MAX_CLAIMS} of {len(raw_claims)} claims were checked")
        raw_claims = raw_claims[:MAX_CLAIMS]
    claims: list[Claim] = []
    for n, raw in enumerate(raw_claims, 1):
        claim = _check_claim(n, raw, passages, problems)
        if claim is not None:
            claims.append(claim)
    if not data["answerable"]:
        if claims:
            problems.append("the answer says the passages don't answer the question, so its claims were ignored")
        return Checked(False, (), tuple(problems))
    return Checked(True, tuple(claims), tuple(problems))


def _check_claim(n: int, raw: Any, passages: Sequence[Hit], problems: list[str]) -> Claim | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not isinstance(raw.get("citations"), list):
        problems.append(f"claim {n} dropped: not an object with text and a list of citations")
        return None
    text = " ".join(raw["text"].split())
    if not text:
        problems.append(f"claim {n} dropped: no text")
        return None
    citations: list[Citation] = []
    raw_citations: list[Any] = raw["citations"][:MAX_CITATIONS]
    for m, rc in enumerate(raw_citations, 1):
        citation = _check_citation(rc, passages)
        if isinstance(citation, str):
            problems.append(f"claim {n}, citation {m} dropped: {citation}")
        elif citation not in citations:
            citations.append(citation)
    if not citations:
        problems.append(f"claim {n} dropped: no citation that checks out")
        return None
    return Claim(text, tuple(citations))


def _check_citation(raw: Any, passages: Sequence[Hit]) -> Citation | str:
    """The citation, or why it doesn't count."""
    if not isinstance(raw, dict):
        return "not an object"
    number, quote = raw.get("passage"), raw.get("quote")
    # type() rather than isinstance: True is an int to Python, and 2.0 isn't a passage number.
    if type(number) is not int or not 1 <= number <= len(passages):
        return f"passage {number!r} isn't one of the {len(passages)} given"
    if not isinstance(quote, str) or len("".join(quote.split())) < MIN_QUOTE_CHARS:
        return f"the quote must hold at least {MIN_QUOTE_CHARS} characters besides spaces"
    if len(quote) > MAX_QUOTE_CHARS:
        return f"the quote is longer than {MAX_QUOTE_CHARS} characters"
    hit = passages[number - 1]
    found = find_quote(quote, hit.content)
    if found is None:
        return f"the quote isn't in passage {number}"
    start, end = found
    return Citation(
        passage=number,
        chunk_id=hit.chunk_id,
        path=hit.path,
        url=hit.url,
        quote=hit.content[start:end],
        start=hit.start + start,
        end=hit.start + end,
    )
