from __future__ import annotations

import math
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recall.embedding import DEFAULT_MODEL, DIMENSION, HashEmbedder, passage_text


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_passage_text_puts_the_heading_path_above_the_text() -> None:
    assert passage_text(("Query Parameters", "Defaults"), "As query...") == "Query Parameters > Defaults\n\nAs query..."
    assert passage_text((), "Before any heading.") == "Before any heading."


def test_hash_embedder_names_its_dimension() -> None:
    assert HashEmbedder().dimension == DIMENSION
    assert HashEmbedder().name == "hash-v1-384"
    assert HashEmbedder(8).name == "hash-v1-8"


@settings(deadline=None)
@given(st.text(max_size=200))
def test_hash_vectors_are_unit_length_and_deterministic(text: str) -> None:
    v = HashEmbedder().embed_query(text)
    assert len(v) == DIMENSION
    assert math.isclose(math.sqrt(dot(v, v)), 1.0, rel_tol=1e-9)
    assert HashEmbedder().embed_passages([text]) == [v]


def test_text_without_tokens_gets_a_fixed_unit_vector() -> None:
    v = HashEmbedder().embed_query("  \n\t ")
    assert v[0] == 1.0
    assert sum(abs(x) for x in v) == 1.0


def test_hashing_ignores_case_and_sees_shared_tokens() -> None:
    e = HashEmbedder()
    q = e.embed_query("query parameters defaults")
    near, far = e.embed_passages(["Defaults QUERY Parameters", "Background tasks run after the response"])
    assert dot(q, near) == pytest.approx(1.0)
    assert dot(q, near) > dot(q, far)


def test_no_passages_no_vectors() -> None:
    assert HashEmbedder().embed_passages([]) == []


def test_the_default_model_is_pinned_and_fits_the_column() -> None:
    assert DEFAULT_MODEL.dimension == DIMENSION
    assert len(DEFAULT_MODEL.revision) == 40
    assert DEFAULT_MODEL.name == f"BAAI/bge-small-en-v1.5@{DEFAULT_MODEL.revision}"


# The real model: run where sentence-transformers is installed (`pip install -e '.[embed]'`); CI's model job sets
# RECALL_REQUIRE_MODEL so that a missing install fails instead of skipping.
def real_embedder():  # type: ignore[no-untyped-def]
    try:
        from recall.embedding import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    except Exception as e:  # ModelUnavailable, or a failed download
        if os.environ.get("RECALL_REQUIRE_MODEL"):
            raise
        pytest.skip(f"no embedding model here: {e}")


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    return real_embedder()


@pytest.mark.model
def test_real_vectors_are_unit_length_and_384_wide(model) -> None:  # type: ignore[no-untyped-def]
    vectors = model.embed_passages(["Query Parameters\n\nDeclare them as function parameters.", "x"])
    assert [len(v) for v in vectors] == [DIMENSION, DIMENSION]
    for v in vectors:
        assert math.isclose(math.sqrt(dot(v, v)), 1.0, rel_tol=1e-5)
    assert model.embed_passages([]) == []


@pytest.mark.model
def test_real_model_ranks_the_passage_about_the_question_first(model) -> None:  # type: ignore[no-untyped-def]
    passages = [
        passage_text(("Query Parameters", "Defaults"), "skip: int = 0, limit: int = 10 give the defaults."),
        passage_text(("Background Tasks",), "Run a function after returning a response, such as sending email."),
        passage_text(("CORS",), "Cross-Origin Resource Sharing lets a frontend on another origin call the API."),
    ]
    vectors = model.embed_passages(passages)
    for question, best in [
        ("how do I give a query parameter a default value?", 0),
        ("send an email after the response is returned", 1),
        ("my frontend on another domain can't call the backend", 2),
    ]:
        q = model.embed_query(question)
        scores = [dot(q, v) for v in vectors]
        assert scores.index(max(scores)) == best, (question, scores)


@pytest.mark.model
def test_real_model_counts_its_own_tokens(model) -> None:  # type: ignore[no-untyped-def]
    # [CLS] and [SEP] around the word pieces.
    assert model.count_tokens("hello") == 3


@pytest.mark.model
def test_real_model_names_its_pinned_revision_and_reads_512_tokens(model) -> None:  # type: ignore[no-untyped-def]
    assert model.name == DEFAULT_MODEL.name
    assert model.dimension == DIMENSION
    assert model._model.max_seq_length == DEFAULT_MODEL.max_seq_length == 512


@pytest.mark.model
def test_real_queries_get_the_prefix_and_passages_dont(model) -> None:  # type: ignore[no-untyped-def]
    q = "how do I declare a query parameter?"
    with_prefix = model.embed_passages([DEFAULT_MODEL.query_prefix + q])[0]
    assert model.embed_query(q) == pytest.approx(with_prefix, abs=1e-6)
    assert dot(model.embed_query(q), model.embed_passages([q])[0]) < 0.999


@pytest.mark.model
def test_a_model_spec_of_the_wrong_dimension_is_refused(model) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from recall.embedding import ModelUnavailable, SentenceTransformerEmbedder

    with pytest.raises(ModelUnavailable, match="gives 384 dimensions, not 512"):
        SentenceTransformerEmbedder(replace(DEFAULT_MODEL, dimension=512))


@pytest.mark.model
def test_the_specs_input_length_is_the_models(model) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from recall.embedding import SentenceTransformerEmbedder

    # bge-small's own default is 512 too, so only another value shows the spec is applied.
    short = SentenceTransformerEmbedder(replace(DEFAULT_MODEL, max_seq_length=16))
    long_text = "query parameters " * 100
    assert short._model.max_seq_length == 16
    truncated, prefix = short.embed_passages([long_text, long_text[:200]])
    assert truncated == pytest.approx(prefix, abs=1e-6)


def test_any_failure_to_load_the_model_is_one_short_line(monkeypatch: pytest.MonkeyPatch) -> None:
    st = pytest.importorskip("sentence_transformers")
    from recall.embedding import ModelUnavailable, SentenceTransformerEmbedder

    def corrupt(*args: object, **kwargs: object) -> object:
        raise ValueError("Error while deserializing header:\n" + "x" * 500)  # what a damaged weights file gives

    monkeypatch.setattr(st, "SentenceTransformer", corrupt)
    with pytest.raises(ModelUnavailable) as e:
        SentenceTransformerEmbedder()
    message = str(e.value)
    assert "\n" not in message
    assert message.startswith(f"can't load {DEFAULT_MODEL.name}: Error while deserializing header: xxx")
    assert len(message) == len(f"can't load {DEFAULT_MODEL.name}: ") + 200
