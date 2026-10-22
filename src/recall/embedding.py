"""Embedding models: the interface ingest and retrieval use, a deterministic stub, and the real model.

DESIGN.md "Embeddings" is the specification. Every vector is unit length, so cosine distance is 1 - dot product.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# The column type is vector(384): every embedder here must produce this many dimensions.
DIMENSION = 384


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Names the model and its exact version; stored with the vectors so two models' vectors never mix."""
        ...

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def passage_text(heading_path: Sequence[str], content: str) -> str:
    """What is embedded for a chunk: its heading path (page title first) above its text.

    The chunk text alone often doesn't say what it's about ("As query parameters are not a fixed part of a path...")
    and its heading line isn't chunk text, so the path supplies the context the rendered page shows above it.
    """
    if not heading_path:
        return content
    return " > ".join(heading_path) + "\n\n" + content


_TOKEN = re.compile(r"\w+|[^\w\s]")


class HashEmbedder:
    """A deterministic embedder with no dependencies, for tests and for trying the pipeline without a model.

    Feature hashing: each token (lowercased, as the chunker counts them) adds +1 or -1 to one of `dimension`
    buckets, both chosen by a hash of the token; the sum is scaled to unit length. Texts that share tokens get
    similar vectors, so tests can predict rankings. It knows nothing of meaning: "car" and "automobile" are as far
    apart as any two words.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hash-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self._dimension
        for m in _TOKEN.finditer(text.lower()):
            h = int.from_bytes(hashlib.blake2b(m.group().encode(), digest_size=8).digest(), "big")
            v[h % self._dimension] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:
            # No tokens, or tokens that cancelled out. Cosine distance to a zero vector is undefined (pgvector
            # returns NaN), so return a fixed unit vector instead: every such text is equally far from the rest.
            v[0] = 1.0
            return v
        return [x / norm for x in v]


@dataclass(frozen=True)
class ModelSpec:
    """A sentence-transformers model pinned to one revision of its Hugging Face repository."""

    repo: str
    revision: str  # a commit of the model repository, so a re-upload can't change the vectors under us
    dimension: int
    max_seq_length: int  # model tokens; longer input is truncated by the model, so ingest checks the fit
    query_prefix: str  # what the model's authors say to put before a search query (passages get none)

    @property
    def name(self) -> str:
        return f"{self.repo}@{self.revision}"


BGE_SMALL = ModelSpec(
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension=384,
    max_seq_length=512,
    query_prefix="Represent this sentence for searching relevant passages: ",
)

DEFAULT_MODEL = BGE_SMALL


class ModelUnavailable(RuntimeError):
    """sentence-transformers isn't installed, or the model can't be loaded."""


class SentenceTransformerEmbedder:
    """The real embedder: a small sentence-transformers model on the CPU, vectors scaled to unit length."""

    def __init__(self, spec: ModelSpec = DEFAULT_MODEL, *, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ModelUnavailable(
                "the embedding model needs sentence-transformers: pip install -e '.[embed]' (or use --embedder hash)"
            ) from e
        self.spec = spec
        self._batch_size = batch_size
        self._model: Any = SentenceTransformer(spec.repo, revision=spec.revision, device="cpu")
        found = self._model.get_embedding_dimension()
        if found != spec.dimension:
            raise ModelUnavailable(f"{spec.name} gives {found} dimensions, not {spec.dimension}")
        self._model.max_seq_length = spec.max_seq_length

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    def count_tokens(self, text: str) -> int:
        """The model's own token count for text, special tokens included: what max_seq_length limits."""
        return len(self._model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.spec.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vectors]
