"""Review of day 3 (cf90395): a recording that can't be written breaks the answer instead of being reported."""

from __future__ import annotations

import json
from pathlib import Path

from recall.answer import answer
from recall.llm import Completion, Recorder, Request
from recall.rerank import OverlapReranker
from recall.retrieval import Hit

CONTENT = "Query parameters can have default values. Set them in the function signature."
PASSAGES = [Hit(1, "p.md", "Page", "https://x/p", ("Page",), 10, 10 + len(CONTENT), CONTENT, 0.5, 1, 1)]


class Live:
    name = "live:test"

    def complete(self, request: Request) -> Completion:
        text = {
            "answerable": True,
            "claims": [{"text": "Yes.", "citations": [{"passage": 1, "quote": "default values"}]}],
        }
        return Completion(json.dumps(text), "model-x")


def test_a_recording_that_cant_be_written_doesnt_break_the_answer(tmp_path: Path):
    """Recorder.complete calls write_recording after the live call; an OSError there (RECALL_RECORDINGS names a file,
    a read-only or full disk) is not a ProviderError, so answer() doesn't catch it: the completion just paid for is
    thrown away and `recall ask` / POST /ask fail with a traceback / 500. Either the answer should still be given
    (with a problem saying the recording failed) or the failure should be a ProviderError and fall back."""
    not_a_directory = tmp_path / "recordings"
    not_a_directory.write_text("a file, not a directory")
    result = answer(
        "Can query parameters have default values?", PASSAGES, Recorder(Live(), not_a_directory), OverlapReranker()
    )
    assert result.method in ("llm", "extractive") and result.problems
