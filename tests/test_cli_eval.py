"""recall eval: the tables, the report file, the gate against a baseline, and bad input as one line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_retrieval import load

from recall import cli, db

QUESTIONS = [
    {
        "id": "q1",
        "question": "Which default values do skip and limit have?",
        "kind": "identifier",
        "gold": [{"page": "query.md", "quote": "skip and limit have default values."}],
    },
    {
        "id": "q2",
        "question": "run something after I reply",
        "kind": "paraphrase",
        "gold": [{"page": "tasks.md", "quote": "sending email"}],
    },
]


@pytest.fixture
def corpus(schema: str, monkeypatch: pytest.MonkeyPatch) -> str:
    with connect_to(schema) as conn:
        db.migrate(conn)
        load(conn)
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    return schema


@pytest.fixture
def questions(tmp_path: Path) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    return path


def run(questions: Path, *more: str) -> int:
    return cli.main(["eval", "--questions", str(questions), "--embedder", "hash", "--reranker", "overlap", *more])


def test_eval_prints_a_table_overall_and_by_kind(corpus: str, questions: Path, capsys: pytest.CaptureFixture[str]):
    assert run(questions) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"2 questions ({questions}), ")
    assert "| system | n | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 | MRR@20 |" in out
    assert "\nidentifier questions:\n" in out and "\nparaphrase questions:\n" in out
    rows = [line for line in out.splitlines() if line.startswith("| full-text |")]
    assert rows[0].startswith("| full-text | 2 | ") and rows[1].startswith("| full-text | 1 | 1.000")


def test_eval_writes_a_report_and_checks_against_it(
    corpus: str, questions: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    report = tmp_path / "out" / "baseline.json"
    assert run(questions, "--json", str(report)) == 0
    body = json.loads(report.read_text())
    assert set(body) == {"settings", "scores", "by_kind", "ranks"}
    assert body["settings"]["questions"] == 2 and body["settings"]["embedder"] == "hash-v1-384"
    assert body["settings"]["reranker"] == "overlap-v1" and body["settings"]["depth"] == 20
    assert body["ranks"]["full-text"]["q1"] == 1
    assert run(questions, "--check", str(report)) == 0
    assert capsys.readouterr().out.endswith(f"no worse than {report}\n")

    # A baseline that was better: the gate fails and says where. (With two questions, recall@k may drop by one
    # question's worth, half, so a recall of 0.5 against 1.0 passes; MRR's slack is 0.01.)
    vector = body["scores"]["vector"]
    vector["mrr"] = vector["mrr"] + 0.02  # above 1 if it was 1: only the comparison matters here
    report.write_text(json.dumps(body))
    assert run(questions, "--check", str(report)) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"\nworse than {report}:\n  vector: MRR ") and len(err.splitlines()) == 3


def test_eval_of_other_settings_than_the_baselines_fails(corpus: str, questions: Path, tmp_path: Path, capsys):
    report = tmp_path / "baseline.json"
    assert run(questions, "--json", str(report)) == 0
    assert run(questions, "--check", str(report), "--rrf-k", "10") == 1
    assert "the settings differ from the baseline's (rrf_k)" in capsys.readouterr().err


def test_eval_of_some_systems_loads_only_what_they_need(corpus: str, questions: Path, monkeypatch, capsys):
    def no_models(kind: str) -> None:
        raise AssertionError("loaded a model")

    monkeypatch.setattr(cli, "make_embedder", no_models)
    assert run(questions, "--systems", "full-text") == 0
    out = capsys.readouterr().out
    assert "| full-text |" in out and "| vector |" not in out


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "No such file or directory"),
        ('{"id": "q1"}\n', "line 1: must be an object with exactly id, question, kind and gold"),
    ],
)
def test_a_bad_question_file_is_one_line(
    corpus: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, message: str
):
    path = tmp_path / "questions.jsonl"
    if content is not None:
        path.write_text(content)
    assert run(path) == 1
    err = capsys.readouterr().err
    assert err.startswith("recall: ") and message in err and len(err.splitlines()) == 1


def test_a_gold_quote_the_database_lacks_is_one_line(corpus: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(QUESTIONS[0] | {"gold": [{"page": "query.md", "quote": "not there"}]}) + "\n")
    assert run(path) == 1
    assert capsys.readouterr().err == "recall: q1: the quote occurs 0 times in query.md, not once\n"


def test_unknown_systems_are_refused(questions: Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        run(questions, "--systems", "full-text,bm25")
    assert (
        "use a comma-separated list of full-text, vector, hybrid, vector+rerank, hybrid+rerank"
        in capsys.readouterr().err
    )


def test_eval_answers_checks_faithfulness_and_reports_it(
    corpus: str, questions: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RECALL_LLM", "none")
    report = tmp_path / "report.json"
    assert run(questions, "--systems", "full-text", "--answers", "--json", str(report)) == 0
    out = capsys.readouterr().out
    assert "\nanswers (ask's defaults: hybrid, reranked, 5 passages; no LLM):\n" in out
    assert "| 2 | extractive 2 | " in out
    answers = json.loads(report.read_text())["answers"]
    assert answers["answers"] == 2 and answers["methods"] == {"extractive": 2}
    # The extractive answer is the passages' own sentences: every claim is supported by construction.
    assert answers["supported_claims"] == 1.0 and answers["fully_supported"] == 1.0
    # The answers aren't gated: a baseline without them still compares.
    assert run(questions, "--systems", "full-text", "--answers", "--check", str(report)) == 0


def test_eval_answers_with_a_provider_that_cant_load_is_one_line(
    corpus: str, questions: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RECALL_LLM", "nonsense")
    assert run(questions, "--answers") == 1
    err = capsys.readouterr().err
    assert err.startswith("recall: ") and len(err.splitlines()) == 1
