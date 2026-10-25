"""Review d2c: the full-text node budget of commit f8f5f9a. Fails on 27dcf48 for the reason in the docstring."""

from __future__ import annotations

import psycopg
from test_retrieval import load

from recall import retrieval
from recall.retrieval import full_text


def query_nodes(conn: psycopg.Connection, question: str) -> int:
    """numnode of the tsquery full-text search builds for question (the `query` CTE of _PHRASES, which both rankings
    use; on day 2 it was a subquery of _FULL_TEXT)."""
    sql = f"WITH {retrieval._PHRASES} SELECT numnode(query) FROM query"
    row = conn.execute(sql, {"question": question, **retrieval._TEXT_LIMITS})
    found = row.fetchone()
    assert found is not None
    return found[0]


def test_the_full_text_query_has_at_most_max_query_nodes_nodes(conn: psycopg.Connection):
    """MAX_QUERY_NODES's comment and DESIGN say words are used "only while the query has at most 256 nodes in all",
    and _FULL_TEXT that words are taken "while the running count of nodes stays within MAX_QUERY_NODES". But the
    running count is sum(numnode(phrase)) only: joining the phrases with ' | ' adds one OR node per phrase after the
    first, which isn't counted. So the query can have up to 256 + 63 = 319 nodes: 51 words of three lexemes
    (5 nodes each, 255) are all kept and make a 305-node query; and 8 phrases of 31 nodes plus "pydantic" (counted 249,
    the case test_words_stop_when_the_query_reaches_its_node_budget relies on) are a 257-node query. Correct: count
    the ' | ' too (sum(numnode(phrase) + 1) - 1, i.e. compare sum(numnode + 1) with MAX_QUERY_NODES + 1), so the
    query built never has more than MAX_QUERY_NODES nodes; or document the bound as phrase nodes plus 63."""
    load(conn)
    triples = " ".join(f"a{i}_b{i}_c{i}" for i in range(51))
    assert query_nodes(conn, triples) <= retrieval.MAX_QUERY_NODES
    longs = " ".join(",".join(f"y{j}x{i}" for i in range(16)) for j in range(8))
    assert query_nodes(conn, longs + " pydantic") <= retrieval.MAX_QUERY_NODES
    # What that means for search: with the ORs counted, "pydantic" doesn't fit after the eight long phrases.
    assert full_text(conn, longs + " pydantic") == []
