# Changelog

## v1.0.0 (2026-10-25)

The first release, built over five days (October 21 to 25, 2026). What a later 1.x keeps: the `recall` command's
subcommands, options and exit codes (0 ok, 1 an error it explains, 2 a usage error); the API's endpoints and their
request and response fields; the migrations, applied in order and never edited (a change is a new one); the question
file's format and the evaluation report's JSON (so `eval/baseline.json` stays comparable); the corpus pin (FastAPI
0.157.0, commit `386a28af`) and the pinned models. The default retrieval mode may change with a new evaluation; a
change is recorded here with its numbers.

### The corpus and storage (day 1)

- `recall fetch` and `recall verify`: FastAPI's English documentation at a pinned commit, checked by the commit id the
  archive declares and the SHA-256 of each of the 421 files used (`corpus/manifest.json`), extracted all or nothing.
- Normalization of FastAPI's Markdown extensions to plain Markdown (includes expanded, admonitions, tabs, comments and
  markup removed), idempotent and one pass in document order; 125 pages.
- Chunking by whole blocks under a token bound (256), with heading paths and anchors; each chunk a verbatim span of its
  normalized page (1,503 chunks).
- PostgreSQL 17 storage: documents and chunks, one-transaction idempotent ingest, numbered migrations under an advisory
  lock; `scripts/pg.sh` for a project-local cluster on port 5544.

### Retrieval (days 2 and 5)

- Local embeddings (`BAAI/bge-small-en-v1.5`, pinned revision, 384 dimensions) of each chunk with its heading path, in
  pgvector with an HNSW index; vectors stored with the model's name, and search refuses another model's vectors.
- Full-text search: a generated, weighted `tsvector` with a GIN index; the question's words as phrases, ORed, within
  fixed limits. Two rankings: ts_rank_cd (day 2) and **BM25** (day 5: Lucene's IDF, k1 = 1.2, b = 0.75, each phrase
  a term, tf counted from the vectors' positions in SQL; `search_length` from migration 003).
- Five modes: `full-text`, `bm25`, `vector`, `hybrid` and `hybrid-bm25` (reciprocal rank fusion, k = 60, 50 candidates
  a list, in one SQL query). **The default is `vector`**, chosen by the day-5 ablation.
- `recall search`.

### Answers and the API (day 3)

- A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`, pinned) over the top 20; answers from the top 5.
- An LLM provider interface: recorded responses by default (keyed by the request's SHA-256), Claude when
  `RECALL_LLM=anthropic`, and `RECALL_RECORD=1` to record; never needed by tests or CI.
- The citation check: each claim's quote must be in the passage it cites, as a whole-word span; the extractive answer
  (the passages' best sentences, each its own span) when there is no model answer.
- `recall ask` and `recall serve` (FastAPI: `POST /ask`, `POST /search`, `GET /healthz`), with validated requests and a
  connection pool no model call holds.

### Evaluation (days 4 and 5)

- `eval/questions.jsonl`: 110 synthetic labelled questions (55 identifier, 55 paraphrase), gold given as quotes and
  resolved to chunks when the evaluation runs.
- `recall eval`: recall@1/3/5/10/20 and MRR@20 for nine systems (the five modes, four of them reranked), overall and
  by kind; paired counts with the exact sign test; lexical faithfulness of the answers; the JSON report.
- The CI gate: `recall eval --check eval/baseline.json` fails on a drop of more than one question's worth or 0.01 MRR,
  or on changed settings.
- `scripts/sweep_fusion.py` and `scripts/sweep_bm25.py`: fusion and BM25 settings on two halves of the questions.

### Checking it

- 660 tests (ruff, `mypy --strict`, pytest with Hypothesis) against a real PostgreSQL, 26 of them on the real models and
  the real corpus; CI on GitHub Actions with four jobs (tests with a pgvector service container, the real corpus, the
  cluster script on Ubuntu, and the real models with the evaluation gate).
- Review agents on every day, their findings kept as tests in `tests/review*/` (README "Reviews").
