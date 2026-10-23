"""Reranking: the order it gives, its checks on a reranker's scores, the stub, and the pinned cross-encoder."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest

from recall.embedding import passage_text
from recall.rerank import DEFAULT_RERANKER, OverlapReranker, rerank
from recall.retrieval import Hit


def hit(chunk_id: int, content: str, heading: tuple[str, ...] = ("Page",)) -> Hit:
    return Hit(chunk_id, "p.md", "Page", "https://x/p/", heading, 0, len(content), content, 0.5, chunk_id, None)


class Fixed:
    """Gives the scores it was made with, and records what it was asked."""

    name = "fixed"

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.asked: list[tuple[str, list[str]]] = []

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        self.asked.append((question, list(texts)))
        return self.scores


HITS = [hit(1, "one"), hit(2, "two"), hit(3, "three"), hit(4, "four")]


def test_rerank_orders_by_score_best_first_and_keeps_the_retrieval_rank():
    ranked = rerank("q", HITS, Fixed([0.1, 0.9, -2.0, 0.5]))
    assert [(r.hit.chunk_id, r.score, r.retrieval_rank) for r in ranked] == [
        (2, 0.9, 2),
        (4, 0.5, 4),
        (1, 0.1, 1),
        (3, -2.0, 3),
    ]


def test_equal_scores_keep_the_retrieval_order():
    assert [r.hit.chunk_id for r in rerank("q", HITS, Fixed([0.5, 0.7, 0.5, 0.7]))] == [2, 4, 1, 3]


def test_k_keeps_the_best_k():
    assert [r.hit.chunk_id for r in rerank("q", HITS, Fixed([0.1, 0.9, -2.0, 0.5]), k=2)] == [2, 4]
    assert rerank("q", [], Fixed([])) == []


def test_each_chunk_is_scored_with_its_heading_path_as_it_was_embedded():
    reranker = Fixed([1.0])
    rerank("how?", [hit(1, "Body text.", ("Page", "Section"))], reranker)
    assert reranker.asked == [("how?", [passage_text(("Page", "Section"), "Body text.")])]


@pytest.mark.parametrize(
    "scores", [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, float("nan"), 0.0, 2.0], [float("inf"), 0, 0, 0]]
)
def test_a_reranker_that_gives_the_wrong_scores_is_an_error(scores: list[float]):
    with pytest.raises(ValueError, match="gave"):
        rerank("q", HITS, Fixed(scores))


def test_the_overlap_stub_scores_the_share_of_question_words():
    stub = OverlapReranker()
    assert stub.score("Query parameter defaults", ["defaults of a QUERY", "nothing", "query parameter defaults"]) == [
        pytest.approx(2 / 3),
        0.0,
        1.0,
    ]
    assert stub.score("?!", ["anything"]) == [0.0]
    assert stub.score("q", []) == []


def test_the_default_reranker_is_pinned():
    assert len(DEFAULT_RERANKER.revision) == 40
    assert DEFAULT_RERANKER.name == f"cross-encoder/ms-marco-MiniLM-L6-v2@{DEFAULT_RERANKER.revision}"
    assert DEFAULT_RERANKER.max_length == 512


# The real cross-encoder: where sentence-transformers is installed; CI's model job requires it.
@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    try:
        from recall.rerank import CrossEncoderReranker

        return CrossEncoderReranker()
    except Exception as e:  # ModelUnavailable, or a failed download
        if os.environ.get("RECALL_REQUIRE_MODEL"):
            raise
        pytest.skip(f"no reranker here: {e}")


@pytest.mark.model
def test_the_real_reranker_puts_the_passage_about_the_question_first(model) -> None:  # type: ignore[no-untyped-def]
    passages = [
        passage_text(("Query Parameters", "Defaults"), "skip: int = 0, limit: int = 10 give the defaults."),
        passage_text(("Background Tasks",), "Run a function after returning a response, such as sending email."),
        passage_text(("CORS",), "Cross-Origin Resource Sharing lets a frontend on another origin call the API."),
    ]
    for question, best in [
        ("how do I give a query parameter a default value?", 0),
        ("send an email after the response is returned", 1),
        ("my frontend on another domain can't call the backend", 2),
    ]:
        scores = model.score(question, passages)
        assert scores.index(max(scores)) == best, (question, scores)


@pytest.mark.model
def test_the_real_reranker_is_the_pinned_revision_and_reads_512_tokens(model) -> None:  # type: ignore[no-untyped-def]
    assert model.name == DEFAULT_RERANKER.name
    assert model._model.max_seq_length == 512
    # [CLS] question [SEP] passage [SEP]
    assert model.count_tokens("hello", "world") == 5
    assert model.score("q", []) == []


@pytest.mark.model
def test_the_specs_max_length_is_applied(model) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from recall.rerank import CrossEncoderReranker

    # The model's own default is 512 too, so only another value shows the spec is applied: past 16 tokens the end of
    # a pair is cut off, so two passages that differ only there score the same.
    short = CrossEncoderReranker(replace(DEFAULT_RERANKER, max_length=16))
    text = "query parameters can have default values, and they are optional."
    pair = [text, text + " Background tasks run after the response."]
    assert model.count_tokens("defaults", text) > 16
    a, b = short.score("defaults", pair)
    assert a == pytest.approx(b, abs=1e-5)
    full_a, full_b = model.score("defaults", pair)
    assert abs(full_a - full_b) > 1e-3


def test_any_failure_to_load_the_reranker_is_one_short_line(monkeypatch: pytest.MonkeyPatch) -> None:
    st = pytest.importorskip("sentence_transformers")
    from recall.embedding import ModelUnavailable
    from recall.rerank import CrossEncoderReranker

    def offline(*args: object, **kwargs: object) -> object:
        raise OSError("We couldn't connect to 'https://huggingface.co'\n" + "x" * 500)

    monkeypatch.setattr(st, "CrossEncoder", offline)
    with pytest.raises(ModelUnavailable) as e:
        CrossEncoderReranker()
    message = str(e.value)
    assert "\n" not in message and message.startswith(f"can't load {DEFAULT_RERANKER.name}: We couldn't connect")
    assert len(message) == len(f"can't load {DEFAULT_RERANKER.name}: ") + 200
