# Recall

Retrieval-augmented question answering over a documentation set, with citations for every claim and an evaluation
harness that measures retrieval and answer faithfulness. PostgreSQL 17 does all the retrieval: pgvector (HNSW) for
meaning, full-text search for exact words, fused by reciprocal rank fusion, then reranked by a small cross-encoder.
Everything runs on a laptop CPU with no API key.

Status: day 1 of 5 (design, corpus, chunking, storage). See [DESIGN.md](DESIGN.md) for the specification.

## License

MIT (see [LICENSE](LICENSE)). The corpus is FastAPI's documentation (MIT, © Sebastián Ramírez); it is downloaded by
`recall fetch`, not committed.
