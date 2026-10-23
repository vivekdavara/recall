"""Rerank retrieved chunks with a cross-encoder, which reads the question and a passage together.

DESIGN.md "Reranking" is the specification. Retrieval scores the question and each passage apart (a vector each, or
word matches); a cross-encoder scores each (question, passage) pair in one pass of the model, which is slower but
sees how the two relate. So it reranks only the top of the fused list.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from recall.embedding import ModelUnavailable, passage_text
from recall.retrieval import Hit

# How many of the retrieved chunks are reranked: the top of the hybrid list. Day 4 measures the choice.
DEFAULT_DEPTH = 20


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text, higher is more relevant; only their order within one call matters."""
        ...


_WORD = re.compile(r"\w+")


class OverlapReranker:
    """A deterministic stand-in for tests: the share of the question's distinct words (lowercased) in the text.

    It knows nothing of meaning, like the hash embedder; tests can predict its order exactly.
    """

    name = "overlap-v1"

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        words = set(_WORD.findall(question.lower()))
        if not words:
            return [0.0 for _ in texts]
        return [len(words & set(_WORD.findall(t.lower()))) / len(words) for t in texts]


@dataclass(frozen=True)
class CrossEncoderSpec:
    """A sentence-transformers cross-encoder pinned to one revision of its Hugging Face repository."""

    repo: str
    revision: str
    max_length: int  # model tokens for question and passage together; longer pairs lose the passage's end

    @property
    def name(self) -> str:
        return f"{self.repo}@{self.revision}"


MINILM_L6 = CrossEncoderSpec(
    repo="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    max_length=512,
)

DEFAULT_RERANKER = MINILM_L6


class CrossEncoderReranker:
    """The real reranker: a small cross-encoder trained on MS MARCO passage ranking, on the CPU."""

    def __init__(self, spec: CrossEncoderSpec = DEFAULT_RERANKER, *, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ModelUnavailable("the reranker needs sentence-transformers: pip install -e '.[embed]'") from e
        self.spec = spec
        self._batch_size = batch_size
        try:
            self._model: Any = CrossEncoder(spec.repo, revision=spec.revision, device="cpu", max_length=spec.max_length)
        except Exception as e:  # offline and not cached, a corrupt cache, a revision that's gone
            reason = " ".join(str(e).split())
            raise ModelUnavailable(f"can't load {spec.name}: {reason[:200]}") from e

    @property
    def name(self) -> str:
        return self.spec.name

    def count_tokens(self, question: str, text: str) -> int:
        """The model's tokens for the pair, special tokens included: what max_length limits."""
        return len(self._model.tokenizer(question, text, truncation=False)["input_ids"])

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(
            [(question, t) for t in texts], batch_size=self._batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return [float(s) for s in scores]


@dataclass(frozen=True)
class Reranked:
    hit: Hit
    score: float  # the reranker's
    retrieval_rank: int  # 1-based rank in the list that was reranked


def rerank(question: str, hits: Sequence[Hit], reranker: Reranker, k: int | None = None) -> list[Reranked]:
    """The hits ordered by the reranker's score, best first, the first k (all by default). Equal scores keep the
    retrieval order. A chunk is scored with its heading path above it, as it was embedded."""
    scores = reranker.score(question, [passage_text(h.heading_path, h.content) for h in hits])
    if len(scores) != len(hits):
        raise ValueError(f"{reranker.name} gave {len(scores)} scores for {len(hits)} passages")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError(f"{reranker.name} gave a score that isn't a finite number")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    ranked = [Reranked(hits[i], scores[i], i + 1) for i in order]
    return ranked if k is None else ranked[:k]
