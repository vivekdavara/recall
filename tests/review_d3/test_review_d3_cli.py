"""Review of day 3 (cf90395): `recall ask` failures that are tracebacks instead of one-line errors."""

from __future__ import annotations

import sys
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
    monkeypatch.setenv("RECALL_LLM", "none")
    return schema


STUBS = ["--embedder", "hash", "--reranker", "overlap"]


def one_line_error(capsys: pytest.CaptureFixture[str]) -> str:
    err = capsys.readouterr().err
    assert err.startswith("recall: ") and err.count("\n") == 1, err
    return err


def test_more_passages_than_retrieval_allows_without_the_reranker_is_one_line(
    corpus: str, capsys: pytest.CaptureFixture[str]
):
    """--passages is only `positive`; with --no-rerank it becomes retrieval's k, and k over MAX_K (1000) raises
    ValueError("k and candidates must be from 1 to 1000 ...") out of main as a traceback. It should be one line and
    exit 1, like the passages > depth check (or argparse should bound --passages as it bounds --depth)."""
    try:
        code = cli.main(["ask", "default values", "--no-rerank", "--passages", "1001", "--mode", "full-text", *STUBS])
    except SystemExit as e:  # argparse refusing it is fine too
        assert e.code == 2
        return
    assert code == 1
    one_line_error(capsys)


def test_a_question_holding_a_nul_is_one_line(corpus: str, capsys: pytest.CaptureFixture[str]):
    """cmd_ask refuses lone surrogates but not NUL: psycopg.DataError ("PostgreSQL text fields cannot contain NUL")
    escapes main as a traceback. `recall search` has the same gap. It should be one line and exit 1."""
    assert cli.main(["ask", "default \x00 values", "--mode", "full-text", *STUBS]) == 1
    one_line_error(capsys)


def test_the_live_provider_without_its_package_is_one_line(
    corpus: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """RECALL_LLM=anthropic without the anthropic package: AnthropicProvider raises ProviderError("the live provider
    needs the anthropic package: pip install -e '.[llm]'"), but cmd_ask catches only ValueError around
    provider_from_environment and main doesn't catch ProviderError, so the install hint arrives as a traceback.
    It should be that one line, exit 1."""
    monkeypatch.setenv("RECALL_LLM", "anthropic")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic -> ImportError
    assert cli.main(["ask", "default values", "--mode", "full-text", *STUBS]) == 1
    assert "pip install" in one_line_error(capsys)
