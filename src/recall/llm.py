"""The LLM provider interface: one request in, one completion out. Recorded by default; live only when asked.

DESIGN.md "Answers" is the specification. Tests and CI replay recordings (or use no model at all), so they never
need a key or the network. The live provider runs only when RECALL_LLM=anthropic, with the key from the
environment; RECALL_RECORD=1 then saves each completion as a recording to replay later.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

# What a recording written by hand for a test says it came from. Such a recording exercises the pipeline; it is
# never reported as what a model would answer.
HAND_WRITTEN = "hand-written"

DEFAULT_RECORDINGS = "recordings"
PROVIDERS = ("recorded", "anthropic", "none")


@dataclass(frozen=True)
class Request:
    system: str
    prompt: str
    schema: Mapping[str, Any]  # the JSON schema the response must follow (structured output)

    def key(self) -> str:
        """SHA-256 of a canonical JSON encoding of all three inputs: what a recording is filed under.

        The schema is part of the key: changing it can change the answer as much as changing the prompt.
        """
        canonical = json.dumps(
            {"system": self.system, "prompt": self.prompt, "schema": self.schema}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Completion:
    text: str
    model: str  # the model that wrote the text, as the provider reported it (HAND_WRITTEN for test recordings)
    input_tokens: int = 0
    output_tokens: int = 0
    notes: tuple[str, ...] = ()  # anything the caller should report with the answer (a recording that failed)


class Provider(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, request: Request) -> Completion: ...


class ProviderError(RuntimeError):
    """The provider couldn't produce a completion. The caller answers without the model (the extractive fallback)."""


class RecordingMissing(ProviderError):
    """No recording for this request: the prompt changed, or it was never recorded."""


class ProviderRefused(ProviderError):
    """The model declined the request; whatever it wrote is not an answer."""


class ProviderTruncated(ProviderError):
    """The model hit its output limit, so its JSON may be cut off."""


def _count(value: Any) -> int:
    # Token counts come from vendor JSON and from files on disk: anything but a non-negative int is 0.
    return value if type(value) is int and value >= 0 else 0


def write_recording(directory: Path, request: Request, completion: Completion, provider: str) -> Path:
    """Save `completion` as the answer to `request`; returns the file written."""
    directory.mkdir(parents=True, exist_ok=True)
    key = request.key()
    record = {
        "key": key,
        "provider": provider,
        "model": completion.model,
        "system": request.system,
        "prompt": request.prompt,
        "schema": request.schema,
        "response": completion.text,
        "usage": {"input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens},
    }
    path = directory / f"{key}.json"
    # Written beside the target and renamed, so a failure never leaves half a recording under the real name. Each
    # writer has a file of its own: with one shared ".partial" name, concurrent writers of the same recording renamed
    # each other's half-written files (the day-3 review).
    fd, name = tempfile.mkstemp(dir=directory, prefix=f"{key}.", suffix=".partial")
    partial = Path(name)
    try:
        os.fchmod(fd, 0o644)  # mkstemp's 0600 would make a recording unreadable to anyone else
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2) + "\n")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def load_recording(path: Path) -> tuple[Request, Completion]:
    """Read a recording and check that the request stored in it still hashes to its file name."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        request = Request(record["system"], record["prompt"], record["schema"])
        usage = record.get("usage") or {}
        completion = Completion(
            record["response"],
            record["model"],
            _count(usage.get("input_tokens")),
            _count(usage.get("output_tokens")),
        )
        if not all(isinstance(v, str) for v in (request.system, request.prompt, completion.text, completion.model)):
            raise TypeError("system, prompt, response and model must be strings")
        if not isinstance(request.schema, dict):
            raise TypeError("schema must be an object")
        key = request.key()
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
        raise ProviderError(f"unreadable recording {path.name}: {e}") from None
    if key != path.stem or record.get("key") != path.stem:
        raise ProviderError(f"recording {path.name} doesn't match its request (was it edited?)")
    return request, completion


class RecordedProvider:
    """Answers only the requests it has a recording for; any other is a RecordingMissing."""

    name = "recorded"

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def complete(self, request: Request) -> Completion:
        path = self.directory / f"{request.key()}.json"
        if not path.is_file():
            raise RecordingMissing(f"no recording for this question's prompt in {self.directory}")
        return load_recording(path)[1]


class Recorder:
    """Wraps a live provider and saves every completion it returns, to replay later."""

    def __init__(self, inner: Provider, directory: Path) -> None:
        self.inner = inner
        self.directory = directory
        self.written: list[Path] = []

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, request: Request) -> Completion:
        completion = self.inner.complete(request)
        try:
            self.written.append(write_recording(self.directory, request, completion, self.inner.name))
        except OSError as e:
            # The completion is paid for and good: answer with it, and say it wasn't saved.
            reason = " ".join(str(e).split())[:200]
            return replace(completion, notes=(*completion.notes, f"the answer wasn't recorded: {reason}"))
        return completion


DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
# Server-side fallback: a request the model's safety classifiers decline is re-run on the model Anthropic
# recommends for that kind of refusal, inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    """Claude through the official SDK, with structured output (the response is JSON that follows the schema).

    Opt-in: made only when RECALL_LLM=anthropic. The SDK finds the key in the environment (ANTHROPIC_API_KEY, or a
    profile from `ant auth login`); nothing here reads or stores one. `client` is for tests: anything with the SDK's
    `beta.messages.create`.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        client: Any = None,
        max_tokens: int = 16000,
        effort: str = "medium",
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ProviderError("the live provider needs the anthropic package: pip install -e '.[llm]'") from e
            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self._max_tokens = max_tokens
        self._effort = effort

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, request: Request) -> Completion:
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                betas=[_FALLBACK_BETA],
                fallbacks="default",
                system=request.system,
                messages=[{"role": "user", "content": request.prompt}],
                output_config={
                    "effort": self._effort,
                    "format": {"type": "json_schema", "schema": dict(request.schema)},
                },
            )
        except Exception as e:
            # The SDK has already retried what can be retried (connection errors, 408, 409, 429, 5xx). Every failure
            # left means the same thing here, answering without the model, so one class carries the reason.
            reason = " ".join(str(e).split())[:300]
            raise ProviderError(f"{self.name} failed: {type(e).__name__}: {reason}") from e
        if response.stop_reason == "refusal":
            raise ProviderRefused(f"{self.name} declined the request")
        if response.stop_reason == "max_tokens":
            raise ProviderTruncated(f"{self.name} stopped at max_tokens ({self._max_tokens}); the answer is incomplete")
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return Completion(
            text,
            str(response.model),
            _count(getattr(usage, "input_tokens", 0)),
            _count(getattr(usage, "output_tokens", 0)),
        )


def provider_from_environment(environ: Mapping[str, str] | None = None) -> Provider | None:
    """RECALL_LLM: `recorded` (the default), `anthropic` (live) or `none` (always the extractive answer).

    RECALL_RECORDINGS is the recordings directory (default `recordings`), RECALL_LLM_MODEL the live model, and
    RECALL_RECORD=1 saves what the live provider returns.
    """
    env = os.environ if environ is None else environ
    kind = env.get("RECALL_LLM", "recorded")
    recordings = Path(env.get("RECALL_RECORDINGS", DEFAULT_RECORDINGS))
    if kind == "none":
        return None
    if kind == "recorded":
        return RecordedProvider(recordings)
    if kind == "anthropic":
        live = AnthropicProvider(env.get("RECALL_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL))
        return Recorder(live, recordings) if env.get("RECALL_RECORD") == "1" else live
    raise ValueError(f"RECALL_LLM must be one of {', '.join(PROVIDERS)}, not {kind!r}")
