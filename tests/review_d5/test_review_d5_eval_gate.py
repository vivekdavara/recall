"""Review of ac9d86e (BM25 ranking): the evaluation's settings and the baseline gate (evaluation.compare, the CLI's
default systems, the committed eval/baseline.json). Each test fails on ac9d86e for the reason in its docstring."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from recall import cli, retrieval
from recall.evaluation import SYSTEMS, compare

BASELINE = Path(__file__).resolve().parents[2] / "eval" / "baseline.json"


def committed_baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_the_committed_baseline_gates_the_default_eval_run():
    """Defect: CI's eval gate fails on this commit even if no system got worse.

    .github/workflows/ci.yml runs `recall eval --answers --json eval-report.json --check eval/baseline.json` with the
    default systems, which are now all nine (cli.py:428, default=list(SYSTEMS)), and every report measuring bm25 or
    hybrid-bm25 now carries settings["bm25"] = {"k1": 1.2, "b": 0.75} (evaluation.py:348). eval/baseline.json was not
    regenerated: its settings have no "bm25" key and its scores cover only the five day-4 systems. So compare()
    (evaluation.py:435-441) returns "the settings differ from the baseline's (bm25): write a new baseline" before
    looking at a single score.

    Input: the baseline's own settings plus the bm25 key evaluate() adds for the default systems, and every system's
    scores no worse than the baseline's (the four new systems copy an old system's scores).
    Observed: compare() reports the settings as different, so `recall eval --check` exits 1 and CI fails.
    Expected: the committed baseline is a report of the default run (all of SYSTEMS, with the bm25 setting), so a run
    no worse than it passes. (Fixed by 002676a, the new baseline; this test reads the committed file.)
    """
    defaults = cli.parser().parse_args(["eval"]).systems
    assert defaults == list(SYSTEMS)
    baseline = committed_baseline()
    current = copy.deepcopy(baseline)
    current["settings"]["bm25"] = {"k1": retrieval.BM25_K1, "b": retrieval.BM25_B}
    stand_in = {"bm25": "full-text", "hybrid-bm25": "hybrid", "bm25+rerank": "vector+rerank"}
    stand_in["hybrid-bm25+rerank"] = "hybrid+rerank"
    current["scores"] = {
        s: copy.deepcopy(baseline["scores"].get(s) or baseline["scores"][stand_in[s]]) for s in defaults
    }
    assert compare(current, baseline) == []


def test_a_settings_key_the_baseline_lacks_is_named_when_the_settings_differ():
    """Defect: a run configured exactly like a day-4 baseline fails the gate with an empty list of changed settings.

    evaluate() now always writes a "bm25" settings key, None when no BM25 system is measured (evaluation.py:348).
    A baseline written before this commit has no such key. compare() decides the settings differ with
    `current["settings"] != baseline["settings"]` (evaluation.py:435), where a None value and a missing key differ,
    but lists the changed keys with `.get(k)` on both sides (evaluation.py:436-440), where they are equal.

    Input: the committed eval/baseline.json, and a report of the same five systems with the same scores and settings
    plus "bm25": None (what `recall eval --systems full-text,vector,hybrid,vector+rerank,hybrid+rerank` now writes).
    Observed: ["the settings differ from the baseline's (): write a new baseline"]: a failure naming nothing.
    Expected: the two checks agree: either no settings problem (absent and None both mean "not measured"), or a
    message that names bm25.
    """
    baseline = committed_baseline()
    # The committed baseline has the key since 002676a; a day-4 one didn't.
    del baseline["settings"]["bm25"]
    current = copy.deepcopy(baseline)
    current["settings"]["bm25"] = None
    problems = compare(current, baseline)
    assert problems == [] or "bm25" in problems[0], problems
