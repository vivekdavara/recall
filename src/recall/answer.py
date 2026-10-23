"""Answer a question: retrieve, rerank, then ask the LLM for cited claims, or fall back to the extractive answer.

DESIGN.md "Answers" is the specification. Whatever writes the answer, every claim that reaches the caller cites a
span of a retrieved passage: the model's claims go through the citation check, and the extractive answer is made of
the passages' own sentences.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import psycopg

from recall import retrieval
from recall.citations import ANSWER_SCHEMA, Claim, MalformedAnswer, check
from recall.embedding import Embedder
from recall.extractive import DEFAULT_SENTENCES, extract
from recall.llm import Provider, ProviderError, Request
from recall.rerank import DEFAULT_DEPTH, Reranker, rerank
from recall.retrieval import Hit

# How many passages an answer is written from: the top of the reranked list.
DEFAULT_PASSAGES = 5

SYSTEM_PROMPT = (
    "You answer questions about FastAPI using only the numbered passages from its documentation that you are given. "
    "Answer in a few short claims. Each claim cites the passages that support it: the passage's number, and a quote "
    "copied word for word from that passage (a phrase or a sentence, not a paraphrase). Say nothing the passages "
    "don't support. If the passages don't answer the question, set answerable to false and give no claims."
)

Method = Literal["llm", "extractive", "none"]


@dataclass(frozen=True)
class Answer:
    question: str
    # llm: the model's claims that passed the check; extractive: the passages' own sentences; none: nothing to
    # answer from (no passages, or the model said they don't answer the question).
    method: Method
    claims: tuple[Claim, ...]
    passages: tuple[Hit, ...]
    provider: str | None  # the provider asked, if any (even when its answer wasn't used)
    model: str | None  # the model whose claims these are (llm only)
    problems: tuple[str, ...] = ()  # why the model's answer, or parts of it, weren't used

    @property
    def text(self) -> str:
        """The claims as one paragraph, each followed by the numbers of the passages it cites: "A. [1] [3] B. [2]"."""
        return " ".join(
            c.text + "".join(f" [{n}]" for n in dict.fromkeys(x.passage for x in c.citations)) for c in self.claims
        )


def build_request(question: str, passages: Sequence[Hit]) -> Request:
    """The prompt for the model: the passages, numbered from 1, each with its page and heading path, then the question.
    Deterministic, so a recording of it can be found again."""
    parts = []
    for n, hit in enumerate(passages, 1):
        where = " > ".join(hit.heading_path)
        parts.append(f"[{n}] {hit.path}" + (f" ({where})" if where else "") + f"\n{hit.content}")
    prompt = "Passages:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"
    return Request(SYSTEM_PROMPT, prompt, ANSWER_SCHEMA)


def answer(
    question: str,
    passages: Sequence[Hit],
    provider: Provider | None,
    reranker: Reranker,
    *,
    documents: Mapping[str, str] | None = None,
    sentences: int = DEFAULT_SENTENCES,
) -> Answer:
    """The model's checked claims if it gives any, else the extractive answer. `documents` (path to page text) lets
    the extractive answer tell code from prose in a chunk that starts inside a code block."""
    passages = tuple(passages)
    name = provider.name if provider is not None else None
    if not passages:
        return Answer(question, "none", (), passages, name, None)
    problems: list[str] = []
    if provider is not None:
        try:
            completion = provider.complete(build_request(question, passages))
            checked = check(completion.text, passages)
        except (ProviderError, MalformedAnswer) as e:
            problems.append(str(e))
        else:
            problems.extend(checked.problems)
            if not checked.answerable:
                return Answer(question, "none", (), passages, name, completion.model, tuple(problems))
            if checked.claims:
                return Answer(question, "llm", checked.claims, passages, name, completion.model, tuple(problems))
            problems.append("no claim in the model's answer had a citation that checks out")
    claims = extract(question, passages, reranker, sentences, documents)
    return Answer(question, "extractive" if claims else "none", tuple(claims), passages, name, None, tuple(problems))


def load_documents(conn: psycopg.Connection, paths: Sequence[str]) -> dict[str, str]:
    """The normalized text of the pages at these paths. Read after the search, in a transaction of its own, so a
    page may have changed meanwhile; the extractive answer checks each passage is still in its page."""
    if not paths:
        return {}
    with conn.transaction():
        rows = conn.execute("SELECT path, content FROM documents WHERE path = ANY(%s)", (list(set(paths)),)).fetchall()
    return {path: content for path, content in rows}


@dataclass
class Timings:
    """Milliseconds spent in each step of ask()."""

    retrieve: float = 0.0
    rerank: float = 0.0
    answer: float = 0.0

    @property
    def total(self) -> float:
        return self.retrieve + self.rerank + self.answer


def ask(
    conn: psycopg.Connection,
    question: str,
    embedder: Embedder | None,
    reranker: Reranker,
    provider: Provider | None,
    *,
    mode: retrieval.Mode = "hybrid",
    passages: int = DEFAULT_PASSAGES,
    depth: int = DEFAULT_DEPTH,
    candidates: int = retrieval.DEFAULT_CANDIDATES,
    use_reranker: bool = True,
) -> tuple[Answer, Timings]:
    """Retrieve `depth` chunks (or `passages`, without the reranker), rerank them, and answer from the top `passages`.
    Raises what retrieval.search raises (a model mismatch, a bad size, a database error)."""
    if passages < 1 or (use_reranker and passages > depth):
        # depth is how many are reranked, so without the reranker it doesn't bound passages.
        raise ValueError(f"passages must be from 1 to depth ({depth}), not {passages}")
    timings = Timings()
    t = time.perf_counter()
    hits = retrieval.search(conn, question, embedder, mode, depth if use_reranker else passages, candidates=candidates)
    timings.retrieve = (time.perf_counter() - t) * 1000
    if use_reranker:
        t = time.perf_counter()
        hits = [r.hit for r in rerank(question, hits, reranker, passages)]
        timings.rerank = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    documents = load_documents(conn, [h.path for h in hits])
    result = answer(question, hits, provider, reranker, documents=documents)
    timings.answer = (time.perf_counter() - t) * 1000
    return result, timings
