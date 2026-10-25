"""BM25 full-text ranking (DESIGN.md "Retrieval"): checked against a reference written here in Python from the search
vectors' positions, and its phrase counts against PostgreSQL's own phrase match."""

from __future__ import annotations

import math
from collections.abc import Sequence

import psycopg
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_retrieval import load, where

from recall import retrieval
from recall.corpus.load import Document
from recall.embedding import HashEmbedder
from recall.retrieval import BM25_B, BM25_K1, full_text, hybrid, search


def doc(path: str, text: str, title: str = "Page") -> Document:
    return Document(path, title, f"https://x/{path.removesuffix('.md')}/", f"# {title}\n\n{text}\n")


def phrases(conn: psycopg.Connection, question: str) -> list[tuple[str, list[tuple[str, int]]]]:
    """The question's terms as the reference sees them: each word whose phrase isn't empty, the first word of each
    distinct phrase, with its lexemes and their offsets from to_tsvector. (Questions here are within the limits.)"""
    out: list[tuple[str, list[tuple[str, int]]]] = []
    seen: set[str] = set()
    for word in question.split():
        row = conn.execute(
            "SELECT phraseto_tsquery('english', %(w)s)::text,"
            " (SELECT array_agg(l.lexeme || ' ' || p ORDER BY p) FROM unnest(to_tsvector('english', %(w)s)) l,"
            " unnest(l.positions) p)",
            {"w": word},
        ).fetchone()
        assert row is not None
        phrase, items = row
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        pairs = [(i.rsplit(" ", 1)[0], int(i.rsplit(" ", 1)[1])) for i in items]
        first = pairs[0][1]
        out.append((phrase, [(lexeme, p - first) for lexeme, p in pairs]))
    return out


def positions(conn: psycopg.Connection) -> dict[int, dict[str, set[int]]]:
    rows = conn.execute("SELECT c.id, l.lexeme, l.positions FROM chunks c, unnest(c.search_vector) l").fetchall()
    out: dict[int, dict[str, set[int]]] = {}
    for chunk_id, lexeme, at in rows:
        out.setdefault(chunk_id, {})[lexeme] = set(at)
    return out


def occurrences(where_: dict[str, set[int]], lexemes: Sequence[tuple[str, int]]) -> int:
    """How many start positions have every lexeme at its offset."""
    first, shift = lexemes[0]
    starts = {p - shift for p in where_.get(first, set())}
    return sum(all(s + o in where_.get(lexeme, set()) for lexeme, o in lexemes) for s in starts)


def reference(conn: psycopg.Connection, question: str) -> dict[int, float]:
    """BM25 of every chunk that holds a phrase, from the formula in DESIGN.md."""
    vectors = positions(conn)
    lengths = dict(conn.execute("SELECT id, search_length FROM chunks").fetchall())
    n = len(lengths)
    average = sum(lengths.values()) / n
    scores: dict[int, float] = {}
    for _, lexemes in phrases(conn, question):
        tf = {c: occurrences(v, lexemes) for c, v in vectors.items()}
        df = sum(1 for t in tf.values() if t)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for c, t in tf.items():
            if t:
                norm = t + BM25_K1 * (1 - BM25_B + BM25_B * lengths[c] / average)
                scores[c] = scores.get(c, 0.0) + idf * t * (BM25_K1 + 1) / norm
    return scores


def bm25(conn: psycopg.Connection, question: str, k: int = 1000) -> list[retrieval.Hit]:
    return full_text(conn, question, k, ranking="bm25")


def assert_matches_reference(conn: psycopg.Connection, question: str) -> None:
    hits = bm25(conn, question)
    expected = reference(conn, question)
    assert {h.chunk_id: h.score for h in hits} == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert [h.full_text_rank for h in hits] == list(range(1, len(hits) + 1))


def test_the_search_length_is_the_number_of_lexeme_positions(conn: psycopg.Connection):
    load(conn, [doc("a.md", "Skip the skip and limit, then limit again.", title="Skip")])
    # Title "Skip" (1) + "skip skip limit limit" (4; "the", "and", "then", "again" are stop words).
    rows = conn.execute("SELECT search_length FROM chunks").fetchall()
    assert rows == [(5,)]
    same = conn.execute("SELECT bool_and(search_vector = chunk_search_vector(heading_path, content)) FROM chunks")
    assert same.fetchone() == (True,)


def test_one_word_in_one_chunk_scores_by_the_formula(conn: psycopg.Connection):
    load(conn, [doc("a.md", "alpha beta beta"), doc("b.md", "gamma delta epsilon zeta"), doc("c.md", "eta theta")])
    lengths = dict(
        conn.execute("SELECT d.path, c.search_length FROM chunks c JOIN documents d ON d.id = c.document_id")
    )
    assert lengths == {"a.md": 4, "b.md": 5, "c.md": 3}  # the title "Page" is one lexeme of each
    average = 12 / 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    tf = 2
    expected = idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * 4 / average))
    [hit] = bm25(conn, "beta")
    assert hit.path == "a.md"
    assert hit.score == pytest.approx(expected, rel=1e-12)


def test_a_rare_word_outweighs_a_common_one_where_ts_rank_cd_counts_them_alike(conn: psycopg.Connection):
    common = [doc(f"c{i}.md", "response response response here") for i in range(8)]
    load(conn, [*common, doc("rare.md", "response with lifespan")])
    # Every page says "response" three times; one says it once, with "lifespan", which no other page has.
    assert where(bm25(conn, "response lifespan", 1)) == ["rare/"]
    # ts_rank_cd has no IDF: three "response"s beat one "response" and one "lifespan".
    assert where(full_text(conn, "response lifespan", 1)) != ["rare/"]


def test_a_longer_chunk_is_discounted_for_the_same_count(conn: psycopg.Connection):
    load(conn, [doc("short.md", "cookie value"), doc("long.md", "cookie " + " ".join(f"w{i}" for i in range(30)))])
    hits = bm25(conn, "cookie")
    assert where(hits) == ["short/", "long/"]
    assert hits[0].score > hits[1].score


def test_an_identifier_counts_as_written_not_as_its_parts(conn: psycopg.Connection):
    load(
        conn,
        [
            doc("id.md", "Set response_model_exclude_unset to skip.", title="Settings"),
            doc(
                "parts.md", "A response model can exclude fields, or unset ones; the response model again.", "Response"
            ),
        ],
    )
    assert where(bm25(conn, "response_model_exclude_unset")) == ["id/"]
    assert where(bm25(conn, "the response_model_exclude_unset parameter")) == ["id/"]
    assert sorted(where(bm25(conn, "response model exclude unset"))) == ["id/", "parts/"]


def test_a_phrase_that_repeats_a_lexeme_counts_each_occurrence(conn: psycopg.Connection):
    load(conn, [doc("a.md", "data data data"), doc("b.md", "data value data"), doc("c.md", "other")])
    # data_data is 'data' <-> 'data': twice in a.md (starting at its 1st and 2nd "data"), never in b.md.
    assert where(bm25(conn, "data_data")) == ["a/"]
    assert_matches_reference(conn, "data_data")
    [(_, lexemes)] = phrases(conn, "data_data")
    vectors = positions(conn)
    assert sorted(occurrences(v, lexemes) for v in vectors.values()) == [0, 0, 2]


def test_a_stop_word_inside_a_word_leaves_its_gap(conn: psycopg.Connection):
    load(conn, [doc("gap.md", "skip the limit"), doc("next.md", "skip limit"), doc("far.md", "skip one limit")])
    # skip_the_limit is 'skip' <2> 'limit': the stop word keeps its place, so "skip limit" isn't it, and any word
    # there is ("skip one limit").
    assert sorted(where(bm25(conn, "skip_the_limit"))) == ["far/", "gap/"]
    assert_matches_reference(conn, "skip_the_limit")


def test_a_hyphenated_word_is_its_whole_and_its_parts(conn: psycopg.Connection):
    load(conn, [doc("a.md", "a built-in validator"), doc("b.md", "built in a day")])
    assert where(bm25(conn, "built-in")) == ["a/"]
    assert_matches_reference(conn, "built-in")


def test_words_with_the_same_phrase_count_once(conn: psycopg.Connection):
    load(conn)
    assert bm25(conn, "model models Model") == bm25(conn, "model")


def test_the_reference_agrees_on_the_test_corpus(conn: psycopg.Connection):
    load(conn)
    for question in ["What are the default values?", "background tasks email", "Pydantic JSON model", "CORS origin"]:
        assert_matches_reference(conn, question)


@pytest.mark.parametrize("question", ["", "what is it?", "   ", "!!!", "a & b | !c <-> (d)", "it's \\ 'quoted' :*"])
def test_questions_without_searchable_words_or_with_tsquery_syntax_are_safe(conn: psycopg.Connection, question: str):
    load(conn)
    hits = bm25(conn, question, 10)
    assert len({h.chunk_id for h in hits}) == len(hits)
    assert {h.chunk_id for h in hits} == {h.chunk_id for h in full_text(conn, question, 10)}


def test_bm25_keeps_the_question_limits(conn: psycopg.Connection):
    load(conn)
    filler = " ".join(f"w{i}" for i in range(retrieval.MAX_QUESTION_WORDS))
    assert where(bm25(conn, "pydantic " * 100 + filler)) == ["body/"]
    assert bm25(conn, filler + " pydantic") == []
    huge = ",".join(f"x{i}" for i in range(151))
    assert where(bm25(conn, huge + " pydantic")) == ["body/"]
    longs = [",".join(f"y{j}x{i}" for i in range(16)) for j in range(8)]
    assert bm25(conn, " ".join(longs) + " pydantic") == []


def test_both_rankings_match_the_same_chunks(conn: psycopg.Connection):
    load(conn)
    for question in ["default values", "background", "pydantic email", "send a response after returning"]:
        assert {h.chunk_id for h in bm25(conn, question)} == {h.chunk_id for h in full_text(conn, question, 1000)}


def test_the_bm25_modes_dispatch(conn: psycopg.Connection):
    load(conn)
    e = HashEmbedder()
    q = "default values"
    assert search(conn, q, None, "bm25", 3) == full_text(conn, q, 3, ranking="bm25")
    fused = hybrid(conn, q, e.embed_query(q), 3, candidates=7, ranking="bm25")
    assert search(conn, q, e, "hybrid-bm25", 3, candidates=7) == fused
    # The fusion takes the BM25 list's ranks.
    text_ranks = {h.chunk_id: h.full_text_rank for h in bm25(conn, q, 7)}
    assert all(h.full_text_rank == text_ranks.get(h.chunk_id) for h in fused)


def test_an_unknown_ranking_is_refused(conn: psycopg.Connection):
    with pytest.raises(ValueError, match="ranking"):
        full_text(conn, "x", ranking="tf-idf")  # type: ignore[arg-type]


def test_an_empty_corpus_has_no_hits(conn: psycopg.Connection):
    assert bm25(conn, "anything") == []


# Words from a small vocabulary, so phrases recur: plain words, stop words, identifiers with stop words inside,
# repeats, and hyphenated words.
WORDS = ["skip", "limit", "the", "of", "data", "model", "built-in", "in", "skip_the_limit", "data_data", "model_of"]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    texts=st.lists(st.lists(st.sampled_from(WORDS), min_size=1, max_size=12), min_size=1, max_size=6),
    question=st.lists(st.sampled_from(WORDS), min_size=1, max_size=5),
)
def test_bm25_equals_the_reference_and_its_counts_equal_postgres_phrase_matches(
    conn: psycopg.Connection, texts: list[list[str]], question: list[str]
):
    # Ingest makes the stored corpus exactly these pages, whatever an earlier example left.
    load(conn, [doc(f"p{i}.md", " ".join(words)) for i, words in enumerate(texts)])
    q = " ".join(question)
    assert_matches_reference(conn, q)
    # The reference's phrase counts against PostgreSQL's phrase match, chunk by chunk.
    vectors = positions(conn)
    for phrase, lexemes in phrases(conn, q):
        matching = {r[0] for r in conn.execute("SELECT id FROM chunks WHERE search_vector @@ %s::tsquery", (phrase,))}
        counted = {c for c, v in vectors.items() if occurrences(v, lexemes) > 0}
        assert counted == matching, phrase
