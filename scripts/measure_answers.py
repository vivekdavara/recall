"""Measure answering on the ingested real corpus: each step's latency in-process, or POST /ask over HTTP.

    .venv/bin/python scripts/measure_answers.py [--questions 200] [--warmup 20] [--mode vector]
    .venv/bin/python scripts/measure_answers.py --url http://127.0.0.1:8109 [--concurrency 4]

Needs `recall ingest` with the real model ($RECALL_DATABASE_URL) and the `embed` extra; over HTTP, a running
`recall serve`. The questions are those of measure_retrieval.py: the corpus's section titles, shuffled with a fixed
seed. They are synthetic and easy, for timing only, not a measure of answer quality (day 4's labelled set is).

In-process, after an untimed warm-up of the first `--warmup` questions, for every question: the mode's top 20 (embed
and query; `--mode`, default ask's), the cross-encoder over those 20, the extractive answer from the top 5 (reading
their pages and scoring their sentences), and ask() end to end in that mode, with no LLM (none of these questions'
prompts is recorded, so the recorded provider would only add a file lookup). Over HTTP: each POST /ask's time at the
client, the clients sending one request after another, and the requests a second over the whole run. Nearest-rank
percentiles, in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from measure_retrieval import ms, questions

from recall import db, retrieval
from recall.answer import DEFAULT_PASSAGES, ask, load_documents
from recall.embedding import SentenceTransformerEmbedder
from recall.extractive import extract
from recall.rerank import DEFAULT_DEPTH, CrossEncoderReranker, rerank


def since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def in_process(qs: list[str], warmup: int, mode: retrieval.Mode) -> None:
    embedder = SentenceTransformerEmbedder()
    reranker = CrossEncoderReranker()
    print(f"reranker {reranker.name}: {sum(p.numel() for p in reranker._model.model.parameters()):,} parameters")
    steps: dict[str, list[float]] = {"retrieve": [], "rerank": [], "extractive": [], "ask": []}
    methods: dict[str, int] = {}
    with db.connect() as conn:
        for i, q in enumerate(qs):
            t = time.perf_counter()
            hits = retrieval.search(conn, q, embedder, mode, DEFAULT_DEPTH)
            times = {"retrieve": since(t)}
            t = time.perf_counter()
            top = [r.hit for r in rerank(q, hits, reranker, DEFAULT_PASSAGES)]
            times["rerank"] = since(t)
            t = time.perf_counter()
            extract(q, top, reranker, documents=load_documents(conn, [h.path for h in top]))
            times["extractive"] = since(t)
            t = time.perf_counter()
            result, _ = ask(conn, q, embedder, reranker, None, mode=mode)
            times["ask"] = since(t)
            methods[result.method] = methods.get(result.method, 0) + 1
            if i >= warmup:
                for name, value in times.items():
                    steps[name].append(value)
    print(f"{len(qs) - warmup} questions timed after {warmup} untimed; answers: {methods}")
    print("| step | p50 | p95 | p99 | max |\n|---|---|---|---|---|")
    labels = {
        "retrieve": f"{mode} top {DEFAULT_DEPTH} (embed + query)",
        "rerank": f"cross-encoder over {DEFAULT_DEPTH}",
        "extractive": f"extractive answer from {DEFAULT_PASSAGES} (pages + sentence scores)",
        "ask": "ask() end to end",
    }
    for name, values in steps.items():
        print(f"| {labels[name]} | {ms(values)} |")


def post(url: str, question: str) -> tuple[float, dict[str, Any]]:
    data = json.dumps({"question": question}).encode()
    request = urllib.request.Request(f"{url}/ask", data, {"content-type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read())
    return (time.perf_counter() - start) * 1000, body


def over_http(url: str, qs: list[str], warmup: int, concurrency: int) -> None:
    for q in qs[:warmup]:
        post(url, q)
    timed_qs = qs[warmup:]
    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda q: post(url, q), timed_qs))
    wall = time.perf_counter() - start
    latencies = [t for t, _ in results]
    server = [b["timings_ms"]["total"] for _, b in results]
    print(
        f"{len(timed_qs)} POST /ask after {warmup} untimed, {concurrency} client(s);"
        f" {len(timed_qs) / wall:.1f} a second"
    )
    print("| measured | p50 | p95 | p99 | max |\n|---|---|---|---|---|")
    print(f"| at the client | {ms(latencies)} |")
    print(f"| in the server (timings_ms.total) | {ms(server)} |")


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--questions", type=int, default=200)
    a.add_argument("--warmup", type=int, default=20)
    a.add_argument("--url", help="time POST /ask on this running server instead")
    a.add_argument("--concurrency", type=int, default=1)
    a.add_argument("--mode", choices=retrieval.MODES, default=retrieval.DEFAULT_MODE, help="in-process only")
    args = a.parse_args()
    with db.connect() as conn:
        qs = questions(conn, args.questions + args.warmup)
    if args.url:
        over_http(args.url.rstrip("/"), qs, args.warmup, args.concurrency)
    else:
        in_process(qs, args.warmup, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
