"""The evaluation harness: the question file, gold chunks from quotes, recall@k and MRR, every system, the gate."""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_retrieval import load

from recall.corpus.load import Document
from recall.embedding import HashEmbedder
from recall.evaluation import (
    DEPTH,
    KS,
    SYSTEMS,
    Gold,
    Question,
    QuestionSetError,
    Ranked,
    compare,
    gold_chunks,
    index_settings,
    paired,
    paired_table,
    parse_questions,
    rank_all,
    report,
    score,
    sign_test,
    table,
)
from recall.rerank import OverlapReranker


def line(**fields: Any) -> str:
    row = {"id": "q1", "question": "Which defaults?", "kind": "identifier", "gold": [{"page": "a.md", "quote": "xyz"}]}
    return json.dumps(row | fields)


def test_a_question_file_parses_into_questions():
    text = line() + "\n\n" + line(id="q2", kind="paraphrase", gold=[{"page": "b.md", "quote": "u"}] * 2) + "\n"
    q1, q2 = parse_questions(text)
    assert q1 == Question("q1", "Which defaults?", "identifier", (Gold("a.md", "xyz"),))
    assert q2.kind == "paraphrase" and q2.gold == (Gold("b.md", "u"), Gold("b.md", "u"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no questions"),
        ("{", "line 1: not JSON"),
        ("[]", "exactly id, question, kind and gold"),
        (line(extra=1), "exactly id, question, kind and gold"),
        (line() + "\n" + line(), "not used before"),
        (line(id=""), "not used before"),
        (line(id=7), "not used before"),
        (line(question="  "), "the question must be text"),
        (line(kind="other"), "kind must be one of identifier, paraphrase"),
        (line(gold=[]), "gold must be a non-empty list"),
        (line(gold={"page": "a.md", "quote": "x"}), "gold must be a non-empty list"),
        (line(gold=[{"page": "a.md"}]), "each gold item"),
        (line(gold=[{"page": "a.md", "quote": " "}]), "each gold item"),
        (line(gold=[{"page": 1, "quote": "x"}]), "each gold item"),
    ],
)
def test_a_bad_question_file_is_refused_with_where_and_why(text: str, message: str):
    with pytest.raises(QuestionSetError, match=message):
        parse_questions(text)


def question(qid: str, *gold: tuple[str, str], kind: str = "identifier", text: str = "default values") -> Question:
    return Question(qid, text, kind, tuple(Gold(p, q) for p, q in gold))  # type: ignore[arg-type]


def chunk_ids(conn: psycopg.Connection, path: str) -> list[int]:
    rows = conn.execute(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.path = %s ORDER BY c.ordinal",
        (path,),
    ).fetchall()
    return [r[0] for r in rows]


def test_gold_chunks_are_the_chunks_a_quote_overlaps(conn: psycopg.Connection):
    load(conn)
    defaults, optional = chunk_ids(conn, "query.md")
    gold = gold_chunks(
        conn,
        [
            question("one", ("query.md", "skip and limit have default values.")),
            question("part", ("query.md", "limit have")),
            question("two pages", ("query.md", "make one optional"), ("tasks.md", "sending email")),
        ],
    )
    assert gold["one"] == {defaults} and gold["part"] == {defaults}
    assert gold["two pages"] == {optional, *chunk_ids(conn, "tasks.md")}


def test_a_quote_across_a_chunk_boundary_makes_both_chunks_gold(conn: psycopg.Connection):
    words = " ".join(f"word{i}" for i in range(40))
    load(conn, [Document("long.md", "Long", "https://x/long/", f"# Long\n\n{words}\n")], max_tokens=16)
    ids = chunk_ids(conn, "long.md")
    assert len(ids) == 3
    gold = gold_chunks(conn, [question("q", ("long.md", "word15 word16 word17"))])
    assert gold["q"] == set(ids[:2])


@pytest.mark.parametrize(
    ("gold", "message"),
    [
        (("missing.md", "default values"), "no page missing.md in the database"),
        (("query.md", "nowhere in the page"), "occurs 0 times in query.md"),
        (("query.md", "e"), "occurs [0-9]+ times in query.md"),
        (("query.md", "# Query Parameters"), "in no chunk"),  # a heading line: headings aren't chunk text
    ],
)
def test_a_gold_quote_must_be_in_one_chunked_place(conn: psycopg.Connection, gold: tuple[str, str], message: str):
    load(conn)
    with pytest.raises(QuestionSetError, match=message):
        gold_chunks(conn, [question("q", gold)])


def test_every_bad_gold_quote_is_reported_at_once(conn: psycopg.Connection):
    load(conn)
    bad = [question(f"q{n}", ("missing.md", "x")) for n in range(12)]
    with pytest.raises(QuestionSetError, match=r"q0: .*q9: .*\(and 2 more\)$"):
        gold_chunks(conn, bad)


def ranked(rank: int | None, kind: str = "identifier", qid: str = "q") -> Ranked:
    # Gold chunk 1, placed at `rank` in a list of 20 others.
    others = list(range(100, 100 + DEPTH))
    ids = others if rank is None else [*others[: rank - 1], 1, *others[rank - 1 : DEPTH - 1]]
    return Ranked(question(qid, ("p.md", "x"), kind=kind), frozenset({1}), tuple(ids))


def test_rank_is_the_first_gold_chunks_place():
    assert ranked(3).rank == 3 and ranked(None).rank is None
    two_gold = Ranked(question("q", ("p.md", "x")), frozenset({7, 9}), (5, 9, 7))
    assert two_gold.rank == 2


def test_recall_at_k_and_mrr_by_hand():
    s = score([ranked(1), ranked(3), ranked(None), ranked(2)])
    assert s.questions == 4
    assert s.recall == {1: 0.25, 3: 0.75, 5: 0.75, 10: 0.75, 20: 0.75}
    assert s.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert score([]).questions == 0 and score([]).mrr == 0.0


@given(st.lists(st.one_of(st.none(), st.integers(1, DEPTH)), min_size=1, max_size=50))
def test_recall_grows_with_k_and_mrr_lies_between_recall_at_1_and_at_depth(ranks: list[int | None]):
    s = score([ranked(r) for r in ranks])
    values = [s.recall[k] for k in KS]
    assert values == sorted(values)
    assert s.recall[1] <= s.mrr + 1e-12 and s.mrr <= s.recall[DEPTH] + 1e-12


def test_every_system_ranks_every_question(conn: psycopg.Connection):
    load(conn)
    questions = [
        question("q1", ("query.md", "skip and limit have default values."), text="Which default values?"),
        question("q2", ("tasks.md", "sending email"), kind="paraphrase", text="run something after I reply"),
    ]
    gold = gold_chunks(conn, questions)
    out = rank_all(conn, questions, gold, HashEmbedder(), OverlapReranker())
    assert list(out) == list(SYSTEMS)
    for system, rs in out.items():
        assert [r.question.id for r in rs] == ["q1", "q2"], system
        assert all(len(r.chunk_ids) <= DEPTH and len(set(r.chunk_ids)) == len(r.chunk_ids) for r in rs)
    assert out["full-text"][0].rank == 1
    for system, base in (("hybrid+rerank", "hybrid"), ("vector+rerank", "vector")):
        for plain, reranked in zip(out[base], out[system], strict=True):
            assert sorted(plain.chunk_ids) == sorted(reranked.chunk_ids)  # the reranker only reorders
    # Full-text needs no embedder or reranker.
    only = rank_all(conn, questions, gold, None, None, ["full-text"])
    assert [r.chunk_ids for r in only["full-text"]] == [r.chunk_ids for r in out["full-text"]]


def test_rank_all_refuses_what_it_cant_run(conn: psycopg.Connection):
    load(conn)
    questions = [question("q", ("query.md", "default values"))]
    gold = gold_chunks(conn, questions)
    with pytest.raises(ValueError, match="unknown systems"):
        rank_all(conn, questions, gold, HashEmbedder(), None, ["bm25"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="need an embedder"):
        rank_all(conn, questions, gold, None, None, ["vector"])
    with pytest.raises(ValueError, match="vector\\+rerank, hybrid\\+rerank needs a reranker"):
        rank_all(conn, questions, gold, HashEmbedder(), None, ["vector+rerank", "hybrid+rerank"])


def test_the_report_splits_by_kind_and_keeps_each_rank():
    rs = [ranked(1, qid="a"), ranked(4, "paraphrase", "b"), ranked(None, "paraphrase", "c")]
    r = report({"vector": rs}, {"embedder": "e"})
    assert r.scores["vector"].recall[1] == pytest.approx(1 / 3)
    assert r.by_kind["vector"]["identifier"].recall[1] == 1.0
    assert r.by_kind["vector"]["paraphrase"].recall[5] == 0.5
    assert r.ranks["vector"] == {"a": 1, "b": 4, "c": None}
    body = json.loads(json.dumps(r.to_json()))
    assert body["scores"]["vector"]["recall"]["5"] == pytest.approx(2 / 3)
    assert table(r.scores).splitlines()[2].startswith("| vector | 3 | 0.333 | 0.333 | 0.667 |")


def baseline(recall: float = 0.5, mrr: float = 0.4, n: int = 100, **settings: Any) -> dict[str, Any]:
    return {
        "settings": {"questions_sha256": "abc", "embedder": "e"} | settings,
        "scores": {"hybrid": {"questions": n, "recall": {"1": recall, "10": recall}, "mrr": mrr}},
    }


def test_the_gate_passes_the_same_or_better_and_one_questions_worth_of_noise():
    assert compare(baseline(), baseline()) == []
    assert compare(baseline(0.9, 0.9), baseline()) == []
    assert compare(baseline(0.49, 0.395), baseline()) == []


def test_the_gate_fails_a_drop_of_two_questions_or_of_mrr():
    assert compare(baseline(0.48), baseline()) == [
        "hybrid: recall@1 0.480 < baseline 0.500",
        "hybrid: recall@10 0.480 < baseline 0.500",
    ]
    assert compare(baseline(mrr=0.38), baseline()) == ["hybrid: MRR 0.380 < baseline 0.400"]
    assert compare({"settings": baseline()["settings"], "scores": {}}, baseline()) == ["hybrid: not measured"]


def test_the_gate_compares_only_the_same_questions_and_settings():
    (problem,) = compare(baseline(questions_sha256="def", embedder="other"), baseline())
    assert problem.startswith("the settings differ from the baseline's (embedder, questions_sha256)")


def test_the_sign_test_by_hand():
    assert sign_test(0, 0) == 1.0
    assert sign_test(15, 0) == pytest.approx(2 / 2**15)
    assert sign_test(5, 0) == 0.0625
    assert sign_test(1, 1) == 1.0
    assert sign_test(18, 5) == pytest.approx(0.0106, abs=5e-5)


@given(st.integers(0, 60), st.integers(0, 60))
def test_the_sign_test_is_symmetric_and_a_probability(a: int, b: int):
    assert sign_test(a, b) == sign_test(b, a)
    assert 0 < sign_test(a, b) <= 1


def test_paired_counts_the_questions_only_one_system_finds():
    ranks = {"x": {"q1": 1, "q2": 4, "q3": None, "q4": 2}, "y": {"q1": 3, "q2": 1, "q3": 2, "q4": None}}
    one = paired(ranks, "x", "y", 1)
    assert (one.a_only, one.b_only) == (1, 1) and one.p_value == 1.0
    five = paired(ranks, "x", "y", 5)
    assert (five.a_only, five.b_only) == (1, 1)  # q4 for x, q3 for y; q1 and q2 both
    with pytest.raises(ValueError, match="same questions"):
        paired({"x": {"q1": 1}, "y": {"q2": 1}}, "x", "y", 1)
    table = paired_table(ranks, "y", ks=(1, 5))
    assert table.splitlines() == [
        "| system vs y | @1 | @5 |",
        "|---|---:|---:|",
        "| x | 1 / 1 (p 1.000) | 1 / 1 (p 1.000) |",
    ]


def test_a_database_with_no_corpus_has_no_settings_to_evaluate(conn: psycopg.Connection):
    with pytest.raises(QuestionSetError, match="no corpus: run recall ingest"):
        index_settings(conn)
    load(conn)
    settings = index_settings(conn)
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert count is not None and settings["chunks"] == count[0] > 0
    assert settings["max_tokens"] == 64 and settings["chunker_version"] >= 1 and len(settings["corpus_commit"]) == 40
