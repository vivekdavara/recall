"""Measure retrieval against labelled questions: recall@k and MRR for each way of ranking chunks.

DESIGN.md "Evaluation" is the specification. A question's gold is one or more quotes from the pages that answer it;
its gold chunks are the chunks whose spans overlap a quote, found in the database when the evaluation runs, so the
labels don't depend on chunk ids or on the chunker's settings. Each system ranks the corpus's chunks for every
question; a question counts as found at k if one of its gold chunks is in the system's top k.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psycopg

from recall import retrieval
from recall.embedding import Embedder
from recall.rerank import DEFAULT_DEPTH, Reranker, rerank

Kind = Literal["identifier", "paraphrase"]
KINDS: tuple[Kind, ...] = ("identifier", "paraphrase")
System = Literal["full-text", "vector", "hybrid", "vector+rerank", "hybrid+rerank"]
SYSTEMS: tuple[System, ...] = ("full-text", "vector", "hybrid", "vector+rerank", "hybrid+rerank")
RERANKED: dict[System, System] = {"vector+rerank": "vector", "hybrid+rerank": "hybrid"}
# How far down each system's list a gold chunk is looked for: the reranker's depth, so hybrid+rerank is exactly what
# ask() does (hybrid's top 20, reordered) and vector+rerank what ask(mode="vector") does. MRR is over this depth
# (MRR@20), and recall@20 is the same with and without the reranker by construction: it only reorders.
DEPTH = DEFAULT_DEPTH
KS = (1, 3, 5, 10, 20)
# The gate's slack: a platform's float differences can move a near-tie, so recall@k may drop by one question's worth
# and MRR by this much before the gate fails.
MRR_SLACK = 0.01

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_BASELINE = Path("eval/baseline.json")


class QuestionSetError(ValueError):
    """The question file isn't a valid question set, or its gold quotes can't be found in the database."""


@dataclass(frozen=True)
class Gold:
    page: str  # a page path, e.g. tutorial/query-params.md
    quote: str  # text that occurs exactly once in the page's normalized text


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    kind: Kind
    gold: tuple[Gold, ...]


def parse_questions(text: str) -> list[Question]:
    """One JSON object a line: {"id", "question", "kind", "gold": [{"page", "quote"}, ...]}, nothing else."""
    questions: list[Question] = []
    seen: set[str] = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise QuestionSetError(f"line {n}: not JSON: {e}") from None
        if not isinstance(row, dict) or set(row) != {"id", "question", "kind", "gold"}:
            raise QuestionSetError(f"line {n}: must be an object with exactly id, question, kind and gold")
        qid, question, kind, gold = row["id"], row["question"], row["kind"], row["gold"]
        if not isinstance(qid, str) or not qid or qid in seen:
            raise QuestionSetError(f"line {n}: the id must be a string not used before, not {qid!r}")
        if not isinstance(question, str) or not question.strip():
            raise QuestionSetError(f"{qid}: the question must be text")
        if kind not in KINDS:
            raise QuestionSetError(f"{qid}: kind must be one of {', '.join(KINDS)}, not {kind!r}")
        if not isinstance(gold, list) or not gold:
            raise QuestionSetError(f"{qid}: gold must be a non-empty list")
        items = []
        for g in gold:
            if (
                not isinstance(g, dict)
                or set(g) != {"page", "quote"}
                or not all(isinstance(g[f], str) and g[f].strip() for f in ("page", "quote"))
            ):
                raise QuestionSetError(f"{qid}: each gold item must be {{page, quote}}, both non-empty text")
            items.append(Gold(g["page"], g["quote"]))
        seen.add(qid)
        questions.append(Question(qid, question, kind, tuple(items)))
    if not questions:
        raise QuestionSetError("no questions")
    return questions


def load_questions(path: Path) -> list[Question]:
    return parse_questions(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """The SHA-256 of the question file: a baseline holds it, so it is compared only with results on the same set."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold_chunks(conn: psycopg.Connection, questions: Sequence[Question]) -> dict[str, frozenset[int]]:
    """Each question's gold chunks: the chunks of a gold page whose span overlaps a gold quote. Raises
    QuestionSetError, listing every problem, if a page isn't in the database, a quote isn't in its page exactly once,
    or a quote lies in no chunk (a heading line is in none)."""
    pages = sorted({g.page for q in questions for g in q.gold})
    with conn.transaction():
        documents: dict[str, str] = dict(
            conn.execute("SELECT path, content FROM documents WHERE path = ANY(%s)", (pages,)).fetchall()
        )
        rows = conn.execute(
            "SELECT d.path, c.id, c.start_offset, c.end_offset FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.path = ANY(%s) ORDER BY c.id",
            (pages,),
        ).fetchall()
    spans: dict[str, list[tuple[int, int, int]]] = {}
    for path, chunk_id, start, end in rows:
        spans.setdefault(path, []).append((chunk_id, start, end))
    problems: list[str] = []
    result: dict[str, frozenset[int]] = {}
    for q in questions:
        ids: set[int] = set()
        for g in q.gold:
            content = documents.get(g.page)
            if content is None:
                problems.append(f"{q.id}: no page {g.page} in the database")
                continue
            count = content.count(g.quote)
            if count != 1:
                problems.append(f"{q.id}: the quote occurs {count} times in {g.page}, not once")
                continue
            start = content.index(g.quote)
            end = start + len(g.quote)
            overlapping = {chunk_id for chunk_id, s, e in spans.get(g.page, []) if s < end and e > start}
            if not overlapping:
                problems.append(f"{q.id}: the quote in {g.page} is in no chunk (a heading line?)")
            ids |= overlapping
        result[q.id] = frozenset(ids)
    if problems:
        more = f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""
        raise QuestionSetError("; ".join(problems[:10]) + more)
    return result


@dataclass(frozen=True)
class Ranked:
    """One system's list for one question, best first, and the question's gold chunks."""

    question: Question
    gold: frozenset[int]
    chunk_ids: tuple[int, ...]

    @property
    def rank(self) -> int | None:
        """The 1-based rank of the first gold chunk in the list; None if none is in it."""
        return next((n for n, chunk_id in enumerate(self.chunk_ids, 1) if chunk_id in self.gold), None)


@dataclass(frozen=True)
class Scores:
    questions: int
    recall: Mapping[int, float]  # k -> the share of questions with a gold chunk in the top k
    mrr: float  # the mean over questions of 1 / the first gold chunk's rank (0 when it isn't in the list)

    def to_json(self) -> dict[str, Any]:
        return {"questions": self.questions, "recall": {str(k): v for k, v in self.recall.items()}, "mrr": self.mrr}


def score(ranked: Sequence[Ranked], ks: Sequence[int] = KS) -> Scores:
    if not ranked:
        return Scores(0, {k: 0.0 for k in ks}, 0.0)
    ranks = [r.rank for r in ranked]
    n = len(ranks)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return Scores(n, recall, mrr)


def rank_all(
    conn: psycopg.Connection,
    questions: Sequence[Question],
    gold: Mapping[str, frozenset[int]],
    embedder: Embedder | None,
    reranker: Reranker | None,
    systems: Sequence[System] = SYSTEMS,
    *,
    depth: int = DEPTH,
    candidates: int = retrieval.DEFAULT_CANDIDATES,
    rrf_k: int = retrieval.RRF_K,
) -> dict[System, list[Ranked]]:
    """Every system's top `depth` for every question. Each question is embedded once, for all the systems."""
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown systems {unknown}: use {', '.join(SYSTEMS)}")
    needs_vectors = any(s != "full-text" for s in systems)
    if needs_vectors:
        if embedder is None:
            raise ValueError("vector and hybrid systems need an embedder")
        retrieval.check_model(conn, embedder)
    reranked = [s for s in systems if s in RERANKED]
    if reranked and reranker is None:
        raise ValueError(f"{', '.join(reranked)} needs a reranker")
    out: dict[System, list[Ranked]] = {s: [] for s in systems}
    for q in questions:
        lists: dict[System, list[retrieval.Hit]] = {}
        if "full-text" in systems:
            lists["full-text"] = retrieval.full_text(conn, q.question, depth)
        if needs_vectors:
            assert embedder is not None
            query_vector = embedder.embed_query(q.question)
            if "vector" in systems or "vector+rerank" in systems:
                lists["vector"] = retrieval.vector(conn, query_vector, depth)
            if "hybrid" in systems or "hybrid+rerank" in systems:
                lists["hybrid"] = retrieval.hybrid(
                    conn, q.question, query_vector, depth, candidates=candidates, rrf_k=rrf_k
                )
        for system in reranked:
            assert reranker is not None
            lists[system] = [r.hit for r in rerank(q.question, lists[RERANKED[system]], reranker)]
        for s in systems:
            out[s].append(Ranked(q, gold[q.id], tuple(h.chunk_id for h in lists[s])))
    return out


@dataclass(frozen=True)
class Report:
    settings: Mapping[str, Any]  # the question set's digest, the models, depth, candidates, rrf_k
    scores: Mapping[str, Scores]  # system -> all questions
    by_kind: Mapping[str, Mapping[str, Scores]]  # system -> kind -> that kind's questions
    ranks: Mapping[str, Mapping[str, int | None]]  # system -> question id -> rank of its first gold chunk

    def to_json(self) -> dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "scores": {s: v.to_json() for s, v in self.scores.items()},
            "by_kind": {s: {k: v.to_json() for k, v in kinds.items()} for s, kinds in self.by_kind.items()},
            "ranks": {s: dict(r) for s, r in self.ranks.items()},
        }


def report(ranked: Mapping[System, Sequence[Ranked]], settings: Mapping[str, Any]) -> Report:
    return Report(
        settings,
        {s: score(rs) for s, rs in ranked.items()},
        {s: {k: score([r for r in rs if r.question.kind == k]) for k in KINDS} for s, rs in ranked.items()},
        {s: {r.question.id: r.rank for r in rs} for s, rs in ranked.items()},
    )


def index_settings(conn: psycopg.Connection) -> dict[str, Any]:
    """What the gold chunks and the rankings depend on in the database: the corpus, the chunker, the chunk count."""
    with conn.transaction():
        row = conn.execute(
            "SELECT source_commit, chunker_version, max_tokens, (SELECT count(*) FROM chunks) FROM corpus_settings"
        ).fetchone()
    if row is None:
        raise QuestionSetError("the database has no corpus: run recall ingest")
    return {"corpus_commit": row[0], "chunker_version": row[1], "max_tokens": row[2], "chunks": row[3]}


def evaluate(
    conn: psycopg.Connection,
    questions: Sequence[Question],
    questions_sha256: str,
    embedder: Embedder | None,
    reranker: Reranker | None,
    systems: Sequence[System] = SYSTEMS,
    *,
    candidates: int = retrieval.DEFAULT_CANDIDATES,
    rrf_k: int = retrieval.RRF_K,
) -> Report:
    """Gold chunks, every system's rankings and their scores, with the settings a baseline must share."""
    uses_vectors = any(s != "full-text" for s in systems)
    settings = {
        "questions_sha256": questions_sha256,
        "questions": len(questions),
        **index_settings(conn),
        "embedder": embedder.name if embedder is not None and uses_vectors else None,
        "reranker": reranker.name if reranker is not None and any(s in RERANKED for s in systems) else None,
        "depth": DEPTH,
        "candidates": candidates if "hybrid" in systems or "hybrid+rerank" in systems else None,
        "rrf_k": rrf_k if "hybrid" in systems or "hybrid+rerank" in systems else None,
    }
    gold = gold_chunks(conn, questions)
    ranked = rank_all(conn, questions, gold, embedder, reranker, systems, candidates=candidates, rrf_k=rrf_k)
    return report(ranked, settings)


def table(scores: Mapping[str, Scores], ks: Sequence[int] = KS) -> str:
    """A Markdown table: one row a system, recall@k for each k, then MRR."""
    head = "| system | n | " + " | ".join(f"recall@{k}" for k in ks) + f" | MRR@{DEPTH} |"
    rule = "|---|---:|" + "---:|" * (len(ks) + 1)
    rows = [
        f"| {s} | {v.questions} | " + " | ".join(f"{v.recall[k]:.3f}" for k in ks) + f" | {v.mrr:.3f} |"
        for s, v in scores.items()
    ]
    return "\n".join([head, rule, *rows])


def sign_test(a_only: int, b_only: int) -> float:
    """The exact two-sided sign test's p-value: how likely a split at least this uneven of the questions only one
    system finds would be if each were as likely to fall either way."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / float(2**n))


@dataclass(frozen=True)
class Paired:
    """Two systems on the same questions at one k: the questions only one of them finds in its top k."""

    a: str
    b: str
    k: int
    a_only: int
    b_only: int

    @property
    def p_value(self) -> float:
        return sign_test(self.a_only, self.b_only)


def paired(ranks: Mapping[str, Mapping[str, int | None]], a: str, b: str, k: int) -> Paired:
    """Compare two systems question by question (a report's ranks). Paired, because both answered the same
    questions: a difference in recall@k is only the questions that one finds and the other misses."""

    def found(system: str, qid: str) -> bool:
        rank = ranks[system][qid]
        return rank is not None and rank <= k

    questions = list(ranks[a])
    if set(questions) != set(ranks[b]):
        raise ValueError(f"{a} and {b} weren't measured on the same questions")
    a_only = sum(found(a, q) and not found(b, q) for q in questions)
    b_only = sum(found(b, q) and not found(a, q) for q in questions)
    return Paired(a, b, k, a_only, b_only)


def paired_table(ranks: Mapping[str, Mapping[str, int | None]], against: str, ks: Sequence[int] = (1, 5, 10)) -> str:
    """A Markdown table: every other system against `against`, at each k, as "found only by it / only by `against`
    (the sign test's p)"."""
    head = f"| system vs {against} | " + " | ".join(f"@{k}" for k in ks) + " |"
    rule = "|---|" + "---:|" * len(ks)
    rows = []
    for system in ranks:
        if system == against:
            continue
        cells = [paired(ranks, system, against, k) for k in ks]
        rows.append(f"| {system} | " + " | ".join(f"{c.a_only} / {c.b_only} (p {c.p_value:.3f})" for c in cells) + " |")
    return "\n".join([head, rule, *rows])


def compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Why `current` (a report's JSON) is worse than `baseline`; empty if it isn't. Both must be of the same question
    set with the same settings, or there's nothing to compare: that is reported too. recall@k may drop by one
    question's worth and MRR by MRR_SLACK (see there)."""
    if current["settings"] != baseline["settings"]:
        changed = sorted(
            k
            for k in set(current["settings"]) | set(baseline["settings"])
            if current["settings"].get(k) != baseline["settings"].get(k)
        )
        return [f"the settings differ from the baseline's ({', '.join(changed)}): write a new baseline"]
    problems = []
    for system, base in baseline["scores"].items():
        now = current["scores"].get(system)
        if now is None:
            problems.append(f"{system}: not measured")
            continue
        slack = 1 / base["questions"] + 1e-9
        for k, value in base["recall"].items():
            if now["recall"].get(k, 0.0) < value - slack:
                problems.append(f"{system}: recall@{k} {now['recall'].get(k, 0.0):.3f} < baseline {value:.3f}")
        if now["mrr"] < base["mrr"] - MRR_SLACK:
            problems.append(f"{system}: MRR {now['mrr']:.3f} < baseline {base['mrr']:.3f}")
    return problems
