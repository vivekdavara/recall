"""The provider interface: request keys, recordings, the recorder, the live provider's request and its failures."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from recall import llm
from recall.llm import (
    HAND_WRITTEN,
    AnthropicProvider,
    Completion,
    ProviderError,
    ProviderRefused,
    ProviderTruncated,
    RecordedProvider,
    Recorder,
    RecordingMissing,
    Request,
)

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
REQUEST = Request("be brief", "what is 2 + 2?", SCHEMA)


def test_the_key_covers_system_prompt_and_schema():
    keys = {
        REQUEST.key(),
        Request("be briefer", REQUEST.prompt, SCHEMA).key(),
        Request(REQUEST.system, "what is 2 + 3?", SCHEMA).key(),
        Request(REQUEST.system, REQUEST.prompt, {**SCHEMA, "required": []}).key(),
    }
    assert len(keys) == 4
    # The same inputs in another key order are the same request.
    assert Request(REQUEST.system, REQUEST.prompt, dict(reversed(list(SCHEMA.items())))).key() == REQUEST.key()


def test_a_recording_replays_and_anything_else_is_missing(tmp_path: Path):
    llm.write_recording(tmp_path, REQUEST, Completion('{"a": "4"}', HAND_WRITTEN, 7, 3), "test")
    provider = RecordedProvider(tmp_path)
    assert provider.complete(REQUEST) == Completion('{"a": "4"}', HAND_WRITTEN, 7, 3)
    with pytest.raises(RecordingMissing):
        provider.complete(Request(REQUEST.system, "what is 2 + 3?", SCHEMA))
    with pytest.raises(RecordingMissing):
        RecordedProvider(tmp_path / "missing").complete(REQUEST)


def test_a_recording_edited_by_hand_is_refused(tmp_path: Path):
    path = llm.write_recording(tmp_path, REQUEST, Completion("{}", HAND_WRITTEN), "test")
    record = json.loads(path.read_text())
    record["prompt"] = "what is 2 + 3?"
    path.write_text(json.dumps(record))
    with pytest.raises(ProviderError, match="doesn't match"):
        RecordedProvider(tmp_path).complete(REQUEST)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        "{}",
        '{"system": 1, "prompt": "p", "schema": {}, "response": "r", "model": "m"}',
        '{"system": "s", "prompt": "p", "schema": [], "response": "r", "model": "m"}',
        '{"system": "s", "prompt": "p", "schema": {}, "response": null, "model": "m"}',
        "[" * 100_000,
    ],
)
def test_an_unreadable_recording_is_a_provider_error(tmp_path: Path, content: str):
    (tmp_path / f"{REQUEST.key()}.json").write_text(content)
    with pytest.raises(ProviderError, match="unreadable recording"):
        RecordedProvider(tmp_path).complete(REQUEST)


def test_bad_token_counts_in_a_recording_read_as_zero(tmp_path: Path):
    path = llm.write_recording(tmp_path, REQUEST, Completion("{}", HAND_WRITTEN), "test")
    record = json.loads(path.read_text())
    record["usage"] = {"input_tokens": -5, "output_tokens": True}
    path.write_text(json.dumps(record))
    assert RecordedProvider(tmp_path).complete(REQUEST) == Completion("{}", HAND_WRITTEN, 0, 0)


def test_the_recorder_saves_what_the_live_provider_returned(tmp_path: Path):
    class Live:
        name = "live:test"

        def complete(self, request: Request) -> Completion:
            return Completion('{"a": "4"}', "model-x", 11, 2)

    recorder = Recorder(Live(), tmp_path)
    assert recorder.name == "live:test"
    assert recorder.complete(REQUEST) == Completion('{"a": "4"}', "model-x", 11, 2)
    assert recorder.written == [tmp_path / f"{REQUEST.key()}.json"]
    assert RecordedProvider(tmp_path).complete(REQUEST) == Completion('{"a": "4"}', "model-x", 11, 2)
    assert json.loads(recorder.written[0].read_text())["provider"] == "live:test"
    assert not list(tmp_path.glob("*.partial"))


def test_a_live_failure_leaves_no_recording(tmp_path: Path):
    class Failing:
        name = "live:test"

        def complete(self, request: Request) -> Completion:
            raise ProviderError("down")

    with pytest.raises(ProviderError):
        Recorder(Failing(), tmp_path).complete(REQUEST)
    assert list(tmp_path.iterdir()) == []


class FakeMessages:
    """Stands in for the SDK's client.beta.messages: records the call, returns a canned response or raises."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.error = error

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def message(stop_reason: str = "end_turn", text: str = '{"a": "4"}') -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=120, output_tokens=9),
    )


def live(messages: FakeMessages) -> AnthropicProvider:
    return AnthropicProvider(client=SimpleNamespace(beta=SimpleNamespace(messages=messages)))


def test_the_live_request_asks_for_structured_output_with_a_fallback():
    messages = FakeMessages(message())
    assert live(messages).complete(REQUEST) == Completion('{"a": "4"}', "claude-opus-5-5", 120, 9)
    (call,) = messages.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == REQUEST.system
    assert call["messages"] == [{"role": "user", "content": REQUEST.prompt}]
    assert call["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["max_tokens"] == 16000


def test_a_refusal_and_a_truncated_answer_are_errors_not_answers():
    with pytest.raises(ProviderRefused):
        live(FakeMessages(message("refusal", ""))).complete(REQUEST)
    with pytest.raises(ProviderTruncated):
        live(FakeMessages(message("max_tokens", '{"a": '))).complete(REQUEST)


def test_any_client_failure_is_one_provider_error_on_one_line():
    error = RuntimeError("connection\nreset\n" + "x" * 1000)
    with pytest.raises(ProviderError) as info:
        live(FakeMessages(error=error)).complete(REQUEST)
    text = str(info.value)
    assert "\n" not in text and text.startswith("anthropic:claude-opus-5-5 failed: RuntimeError: connection reset")
    assert len(text) < 400


def test_the_environment_chooses_the_provider(tmp_path: Path):
    recorded = llm.provider_from_environment({"RECALL_RECORDINGS": str(tmp_path)})
    assert isinstance(recorded, RecordedProvider) and recorded.directory == tmp_path
    assert llm.provider_from_environment({"RECALL_LLM": "none"}) is None
    assert isinstance(llm.provider_from_environment({}), RecordedProvider)
    with pytest.raises(ValueError, match="RECALL_LLM must be one of"):
        llm.provider_from_environment({"RECALL_LLM": "Anthropic"})


# The real SDK against a local stand-in for the API: what goes over the wire, without a key or the network.


class _Handler(BaseHTTPRequestHandler):
    seen: ClassVar[list[tuple[str, dict[str, str], dict[str, Any]]]] = []
    reply: ClassVar[dict[str, Any]] = {}

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append((self.path, dict(self.headers), body))
        data = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def stand_in() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Handler.seen = []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_the_real_sdk_sends_the_request_this_provider_means(stand_in: str):
    anthropic = pytest.importorskip("anthropic")
    _Handler.reply = {
        "id": "msg_local",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": '{"a": "4"}'}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 31, "output_tokens": 6},
    }
    # A placeholder, not a key: the stand-in never checks it, and nothing leaves this machine.
    client = anthropic.Anthropic(api_key="placeholder", base_url=stand_in, max_retries=0)
    completion = AnthropicProvider(client=client).complete(REQUEST)
    assert completion == Completion('{"a": "4"}', "claude-opus-5-5", 31, 6)
    ((path, headers, body),) = _Handler.seen
    assert path.startswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert body["system"] == REQUEST.system
    assert body["messages"] == [{"role": "user", "content": REQUEST.prompt}]


def test_concurrent_writes_of_one_recording_dont_collide(tmp_path: Path):
    # The day-3 review: every writer of a recording used one ".partial" name, so one writer's rename could take the
    # file another was writing (a half-written recording) or leave it nothing to rename (FileNotFoundError).
    completion = Completion('{"a": "4"}', "model-x", 11, 2)
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def write() -> None:
        start.wait()
        for _ in range(25):
            try:
                llm.write_recording(tmp_path, REQUEST, completion, "live:test")
            except BaseException as e:
                errors.append(e)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert RecordedProvider(tmp_path).complete(REQUEST) == completion
    assert [p.name for p in tmp_path.iterdir()] == [f"{REQUEST.key()}.json"]
