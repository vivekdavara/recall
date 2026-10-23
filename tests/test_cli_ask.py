"""recall ask and recall serve: the answer printed, the JSON form, and the errors as one line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import TEST_DATABASE_URL, connect_to
from test_retrieval import load

from recall import cli, db


@pytest.fixture
def corpus(schema: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    with connect_to(schema) as conn:
        db.migrate(conn)
        load(conn)
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    monkeypatch.setenv("RECALL_RECORDINGS", str(tmp_path / "recordings"))
    monkeypatch.delenv("RECALL_LLM", raising=False)
    return schema


ASK = ["ask", "Which default values do skip and limit have?", "--embedder", "hash", "--reranker", "overlap"]


def test_ask_prints_the_answer_its_citations_and_how_it_was_made(corpus: str, capsys: pytest.CaptureFixture[str]):
    assert cli.main([*ASK, "--passages", "2"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("skip and limit have default values. [1]")
    assert "citations:" in lines and any(
        line.startswith("  [1] https://x/query/#defaults  (query.md, chars ") for line in lines
    )
    assert "answered by: extractive; 2 passages" in lines
    assert "  note: no recording for this question's prompt in" in out
    assert any(line.startswith("time: retrieve ") for line in lines)


def test_ask_json_is_what_the_api_returns(corpus: str, capsys: pytest.CaptureFixture[str]):
    assert cli.main([*ASK, "--json", "--no-rerank"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["method"] == "extractive" and body["timings_ms"]["rerank"] == 0.0
    assert set(body) == {
        "question",
        "answer",
        "method",
        "claims",
        "passages",
        "provider",
        "model",
        "problems",
        "timings_ms",
    }


def test_ask_with_no_provider_has_no_note(
    corpus: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RECALL_LLM", "none")
    assert cli.main(ASK) == 0
    assert "note:" not in capsys.readouterr().out


def test_ask_with_an_unknown_provider_is_one_line(
    corpus: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RECALL_LLM", "gpt")
    assert cli.main(ASK) == 1
    assert capsys.readouterr().err == "recall: RECALL_LLM must be one of recorded, anthropic, none, not 'gpt'\n"


def test_ask_full_text_loads_no_embedder(
    corpus: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cli, "make_embedder", lambda kind: pytest.fail("loaded a model"))
    assert cli.main([*ASK, "--mode", "full-text"]) == 0
    assert "default values" in capsys.readouterr().out


def test_ask_refuses_another_models_vectors_before_loading_one(
    corpus: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cli, "make_embedder", lambda kind: pytest.fail("loaded a model"))
    assert cli.main(["ask", "q", "--embedder", "model", "--reranker", "overlap"]) == 1
    assert "the stored vectors are from hash-v1-384" in capsys.readouterr().err


def test_ask_before_any_ingest_says_to_ingest(
    schema: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RECALL_DATABASE_URL", f"{TEST_DATABASE_URL}?options=-csearch_path%3D{schema},public")
    assert cli.main([*ASK, "--mode", "full-text"]) == 1
    assert "has no corpus or an older schema: run recall ingest" in capsys.readouterr().err


def test_ask_with_more_passages_than_depth_is_one_line(corpus: str, capsys: pytest.CaptureFixture[str]):
    assert cli.main([*ASK, "--passages", "6", "--depth", "5"]) == 1
    assert capsys.readouterr().err == "recall: passages must be from 1 to depth (5), not 6\n"


def test_an_unknown_reranker_in_the_environment_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("RECALL_RERANKER", "Model")
    with pytest.raises(SystemExit) as e:
        cli.main(["ask", "q"])
    assert e.value.code == 2 and "RECALL_RERANKER must be one of model, overlap" in capsys.readouterr().err


def test_serve_runs_the_app_on_the_address_given(monkeypatch: pytest.MonkeyPatch):
    uvicorn = pytest.importorskip("uvicorn")
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw, app=app))
    assert cli.main(["serve", "--port", "8199"]) == 0
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 8199)
    assert {r.path for r in seen["app"].routes} >= {"/ask", "/search", "/healthz"}
