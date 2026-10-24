"""Review of day 4 (713a52a): gold resolution, the gate and the sign test (DESIGN.md "Evaluation")."""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_retrieval import load

from recall import cli, db
from recall.corpus.load import Document
from recall.evaluation import Gold, Question, QuestionSetError, compare, gold_chunks, sign_test


def test_a_quote_that_occurs_twice_overlapping_is_refused(conn: psycopg.Connection):
    """gold_chunks counts occurrences with str.count, which counts non-overlapping ones: "no, no" occurs twice in
    "no, no, no" (at 0 and at 4) but count() says 1, so the quote is accepted and only its first place is gold.
    DESIGN: a quote "no longer in its page exactly once ... fails the run". Correct: QuestionSetError."""
    load(conn, [Document("r.md", "R", "https://x/r/", "# R\n\nThe answer is no, no, no and that is final.\n")])
    with pytest.raises(QuestionSetError, match=r"times in r\.md"):
        gold_chunks(conn, [Question("q", "what?", "identifier", (Gold("r.md", "no, no"),))])


def report(recall1: float) -> dict:
    return {
        "settings": {"questions_sha256": "x", "questions": 10},
        "scores": {"vector": {"questions": 10, "recall": {"1": recall1}, "mrr": recall1}},
    }


@pytest.mark.parametrize(
    "baseline",
    [
        {"settings": {"questions_sha256": "x", "questions": 10}, "scores": {}},
    ],
)
def test_a_baseline_with_no_scores_is_not_a_pass(baseline: dict):
    """compare() loops over the baseline's systems, so a baseline with none (hand-edited, truncated or written by a
    future format) passes any current report, however bad. Correct: a baseline that measures nothing, or misses a
    system the current report has, is a problem to report, not an empty list."""
    assert compare(report(0.0), baseline) != []


@pytest.fixture
def corpus(schema: str, monkeypatch: pytest.MonkeyPatch) -> str:
    with connect_to(schema) as conn:
        db.migrate(conn)
        load(conn)
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    return schema


def test_a_baseline_file_holding_null_fails_the_gate(corpus: str, tmp_path: Path):
    """cmd_eval reads the baseline with json.loads and then gates only `if baseline is not None`: a --check file
    holding JSON null (valid JSON) skips the comparison and exits 0, as if no --check was given. Correct: exit 1,
    the baseline isn't a report."""
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Which default values do skip and limit have?",
                "kind": "identifier",
                "gold": [{"page": "query.md", "quote": "skip and limit have default values."}],
            }
        )
        + "\n"
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text("null\n")
    code = cli.main(["eval", "--questions", str(questions), "--systems", "full-text", "--check", str(baseline)])
    assert code == 1


def test_the_sign_test_works_for_large_question_sets():
    """sign_test divides by float(2**n), which overflows once n (the questions only one system finds) reaches 1024:
    a question set of a few thousand makes paired_table, and so recall eval, crash with OverflowError. Correct: a
    p-value (here 1.0 for an even split, ~0 for a lopsided one)."""
    assert sign_test(600, 600) == pytest.approx(1.0, abs=0.05)
    assert sign_test(1000, 100) < 1e-6
