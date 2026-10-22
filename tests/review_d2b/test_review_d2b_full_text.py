"""Review d2b: the full-text cap of commit 1e29847. Fails on 3a4673c for the reason in the docstring."""

from __future__ import annotations

import psycopg
import pytest
from test_retrieval import load

from recall.retrieval import full_text


@pytest.mark.parametrize("separator", [",", ";"])
def test_one_long_word_does_not_break_full_text_search(conn: psycopg.Connection, separator: str):
    """MAX_QUESTION_WORDS caps the number of whitespace-separated words (64 distinct phrases), not the size of the
    tsquery: phraseto_tsquery turns one whitespace-free word into a phrase of all its lexemes, and the default parser
    splits at punctuation. So a question with no spaces in a list, e.g. a pasted JSON array or CSV line
    '0,1,2,...,19999' (~110 KB, which fits in argv), is one "word" whose phrase has 39,999 nodes; casting the
    string_agg text back to tsquery recurses once per node and fails with StatementTooComplex "stack depth limit
    exceeded" (from about 15,000 items), the finding-D failure the cap was meant to end. Here "pydantic" is the first
    word, well inside the cap. Correct: any question returns hits, so the cap must bound the query's items, not only
    its words (e.g. skip or truncate a phrase over some numnode, or keep words while the running sum of numnode
    stays under a limit)."""
    load(conn)
    question = "pydantic " + separator.join(str(i) for i in range(20000))
    assert [h.path for h in full_text(conn, question)] == ["body.md"]
