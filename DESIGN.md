# Recall: design

Recall answers questions about a documentation set and cites the passages it used. It retrieves passages with
PostgreSQL 17 alone: pgvector (an HNSW index over local embeddings) for meaning, Postgres full-text search for exact
words, and reciprocal rank fusion to combine the two. A small cross-encoder reranks the fused list, and an answer is
written from the top passages with a citation for each claim. An evaluation harness measures retrieval (recall@k,
MRR) and answer faithfulness on labelled questions, and CI fails when retrieval gets worse.

This file is the specification. Day 1 builds the corpus, its normalization and chunking, the storage schema and the
local database; the sections for later days say what they will add and are marked so.

## Goals and non-goals

Goals:

- Every answer is grounded: each claim cites a passage, and each passage points at an exact span of a source page.
- Retrieval quality is measured, not asserted: the eval harness (day 4) gives recall@k and MRR for each retriever,
  and the ablation (day 5) compares them.
- Everything runs on a laptop CPU with no API key: local embeddings, a recorded LLM provider by default.
- The corpus can be rebuilt byte for byte from a pinned source by anyone.

Non-goals: multi-tenant serving, documents that change while being queried, languages other than English, PDF or HTML
input, and a hosted LLM in tests or CI.

## The corpus

**Source.** The English documentation of FastAPI (github.com/fastapi/fastapi, `docs/en/docs/`), released under the
MIT license, at tag `0.157.0`, commit `386a28af2f87af1a32ac26d88f07b7f58ab2dfb1`. It is a good test of retrieval: 156
Markdown pages on one subject, so many passages are near-duplicates in topic and differ in detail (path parameters vs
query parameters vs body parameters), and its questions mix plain words ("what status code does a validation error
get?") with exact identifiers (`Depends`, `HTTPException`, `response_model_exclude_unset`) that dense retrieval alone
tends to miss.

**Not committed.** `recall fetch` downloads the archive of the pinned commit and extracts the pages it uses, the code
files they include (`docs_src/`), and FastAPI's `LICENSE`, into `data/corpus/` (gitignored). The MIT license allows
redistribution, but fetching keeps the repository small and makes the source of every byte explicit.

**Integrity.** The archive's own bytes are not pinned: GitHub generates archives on request and has changed their
compression before, which changes the archive's hash without changing its contents. Instead `fetch` checks two things:

1. The commit id the archive declares (git writes it into the tar's global pax header, `comment=<commit>`) must be
   the pinned commit.
2. Every file it extracts must match `corpus/manifest.json`: the path, the size and the SHA-256 of each of the files
   Recall uses. A file the manifest lists but the archive lacks, or a file whose hash differs, fails the fetch, and
   nothing is written unless every file checks out (extraction goes to a temporary directory that is renamed into
   place at the end).

`recall manifest` regenerates the manifest from an archive; it is run once, when the pinned version changes, and the
diff is reviewed like code.

**Which pages.** The pages under `docs/en/docs/` except these, each for a reason:

| Excluded | Why |
|---|---|
| `release-notes.md` | 724 KB of changelog, half the corpus's bytes; its lines name versions, not behaviour, and would crowd out the pages that explain it. |
| `reference/**` | Generated API reference: each page is a `::: fastapi.X` directive that mkdocstrings expands from docstrings at build time; the Markdown holds almost no text. |
| `_llm-test.md`, `translation-banner.md` | Inputs to the translation tooling, not documentation. |
| `fastapi-people.md`, `external-links.md`, `management.md`, `management-tasks.md`, `newsletter.md` | Generated lists of people and links, and project administration. |

Every exclusion is a path rule in `recall.corpus.source`, tested, and the manifest lists exactly the pages kept.

## Normalization

FastAPI's Markdown uses extensions of its documentation build. `recall.corpus.normalize` turns each page into plain
Markdown that reads the way the rendered page does. The normalized text is the document Recall stores, and every
offset Recall reports (chunk spans, citations) is into it.

Outside fenced code blocks, in this order:

1. **Includes** become fenced code blocks with the included file's text. `{* path ln[a:b,c] hl[...] title[...] *}`
   includes a Python file (only the lines `ln` selects, if given; `hl` and `title` only change how the page is drawn,
   so they are dropped); `{!path!}` and `{!> path!}` include any file, fenced with a language from its extension.
   Paths are relative to `docs/en/`, the documentation build's root, and must resolve inside the extracted corpus. A
   missing file is an error, not a silent gap.
2. **Admonitions** (`/// tip`, `/// note | Technical Details`, `/// warning`, ...) become a bold label line (`**Tip**`,
   `**Technical Details**`) followed by their body; the closing `///` is removed. **Tabs** (`//// tab | Python
   3.10+`) become a label line (`Python 3.10+:`); their bodies are kept.
3. **HTML comments** are removed. **HTML tags** are removed and their text kept (`<dfn title="...">parsing</dfn>`
   becomes `parsing`); inline code spans are left untouched, so `` `<b>` `` survives.
4. **Heading ids** (`## Defaults { #defaults }`) are removed from the heading and kept as the section's anchor, so a
   citation can link to `https://fastapi.tiangolo.com/tutorial/query-params/#defaults`.

Inside fenced code blocks nothing changes: a Python comment starting with `#` is not a heading, and `<b>` in an HTML
template is code. Normalization is deterministic and idempotent (normalizing its own output changes nothing); both are
tested.

## Chunking

A chunk is the unit that is embedded, indexed, retrieved and cited. `recall.chunking` splits a normalized document in
three steps:

1. **Sections.** ATX headings (`#` to `######`, outside fenced code) split the document into sections. Each section
   has a heading path (the titles of the headings above it, outermost first) and an anchor. The heading lines
   themselves are not chunk text; the heading path carries them.
2. **Blocks.** A section's body splits at blank lines into blocks, except that a fenced code block is one block, blank
   lines and all.
3. **Packing.** Blocks are packed greedily, in order, into chunks of at most `max_tokens` tokens. A block bigger than
   `max_tokens` alone is split at line ends, and a line bigger than that at spaces; a chunk is never cut inside a word.

**Size measure.** A token is a run of word characters or one punctuation character (`\w+|[^\w\s]`). It is closer to
the word pieces an embedding model sees than counting words is (`@app.get("/items/{item_id}")` is 14 tokens, not 1),
and it never spans whitespace, so a chunk's count is exactly the sum of its blocks' counts and the bound can be checked
exactly. The default `max_tokens` is 256; day 2 measures what that is in the embedding model's own tokens and adjusts
it if chunks would be truncated.

**No overlap.** Chunks do not overlap. Overlap is a common way to keep context that a cut separates; here cuts fall
at block boundaries, and each chunk carries its page title and heading path, which day 2 prepends to the text it
embeds. Overlap would also make the same sentence appear in two retrieved passages and two citations.

**Invariants** (each has a test, most of them a property test over random Markdown):

- **C1 Verbatim.** A chunk's text is exactly `document[start:end]`, so a citation's span shows what was retrieved.
- **C2 Order.** Chunks are in document order and do not overlap: each starts at or after the previous one's end.
- **C3 Coverage.** Every non-whitespace character of the document is in exactly one chunk, except heading lines.
- **C4 Size.** Every chunk has at least 1 and at most `max_tokens` tokens.
- **C5 Sections.** A chunk lies inside one section, and its heading path and anchor are that section's.
- **C6 Code.** A fenced code block of at most `max_tokens` tokens is never split across chunks.
- **C7 Deterministic.** The same document and settings give the same chunks.

## Storage

PostgreSQL 17 with pgvector holds everything. One database for documents, chunks, vectors and full-text indexes means
one transactional ingest, and day 2's hybrid retrieval is one SQL query instead of a join across two systems.

Day 1's schema (`migrations/001_documents_and_chunks.sql`):

- `documents`: `id`, `path` (unique, e.g. `tutorial/query-params.md`), `title`, `url`, `content` (the normalized
  text), `content_sha256`, `ingested_at`.
- `chunks`: `id`, `document_id` (cascade delete), `ordinal` (unique per document), `heading_path text[]`, `anchor`,
  `start_offset`, `end_offset`, `content`, `token_count`. Check constraints mirror C1 to C4 where SQL can see them:
  `0 <= start_offset < end_offset`, `token_count >= 1`, and `content = substr(document.content, ...)` is checked by
  the ingest tests rather than a trigger.
- `schema_migrations`: which migrations ran, so `recall migrate` applies each file once, in order, in a transaction.

**Ingest** (`recall ingest`) is idempotent. For each document, in one transaction: if a row with the same path and
the same content hash exists, nothing changes; otherwise the document row is inserted or updated and its chunks are
replaced. Pages that are no longer in the corpus are deleted, with their chunks. Running ingest twice leaves the same
rows, and the second run reports every document unchanged. The same chunk settings are recorded with the corpus
(`corpus_settings`), so changing `max_tokens` re-chunks every document rather than mixing sizes.

Day 2 adds (`002_...`): `chunks.embedding vector(384)` with an HNSW index (cosine distance), a generated `tsvector`
column with a GIN index, and the model name and dimension in `corpus_settings`, so vectors from two models never mix.

## The local database

`scripts/pg.sh` runs a project-local PostgreSQL 17 cluster on port 5544 with its data in `.pgdata/` (gitignored):
`start` (runs `initdb` the first time, then `pg_ctl start`; listens on localhost only, trust authentication, so it
must not be exposed), `stop`, `status`, `url` (prints the connection URL), `psql`, and `reset` (stop and delete
`.pgdata`). It creates the `recall` database for the app and the `recall_test` database for the tests, and checks
that the `vector` extension can be created before saying it started. In CI the same role is played by the
`pgvector/pgvector:pg17` service container.

Settings come from the environment: `RECALL_DATABASE_URL` (default `postgresql://localhost:5544/recall`),
`RECALL_TEST_DATABASE_URL` (default `.../recall_test`), `RECALL_DATA_DIR` (default `./data`).

## Architecture

```mermaid
flowchart LR
    subgraph offline["Build the index (day 1, day 2)"]
        A[GitHub archive<br/>pinned commit] -->|recall fetch<br/>commit id + manifest| B[data/corpus]
        B -->|normalize| C[plain Markdown]
        C -->|chunk| D[chunks with spans]
        D -->|recall ingest| E[(PostgreSQL 17<br/>documents, chunks)]
        E -.->|day 2: embed| F[(pgvector HNSW)]
        E -.->|day 2: tsvector| G[(full-text GIN)]
    end
    subgraph online["Answer a question (days 2-3)"]
        Q[question] --> V[vector top-k]
        Q --> K[full-text top-k]
        F --> V
        G --> K
        V --> R[reciprocal rank fusion]
        K --> R
        R --> X[cross-encoder rerank]
        X --> L[answer with citations<br/>recorded LLM or extractive]
        L --> API[FastAPI /ask]
    end
    subgraph eval["Measure (days 4-5)"]
        S[labelled questions<br/>with gold passages] --> M[recall@k, MRR,<br/>faithfulness]
        M --> CI[CI gate]
    end
```

## Later days (plan)

- **Day 2, retrieval.** Embeddings with a small sentence-transformers model on CPU (384 dimensions), stored in
  pgvector with an HNSW index; full-text retrieval with `ts_rank_cd` over a weighted `tsvector` (title and heading path
  weighted above body); hybrid retrieval by reciprocal rank fusion (`score = sum 1 / (k + rank)`, k = 60), with each
  retriever's own top-k kept for the ablation.
- **Day 3, answers.** A provider interface for the LLM (a recorded provider by default, so tests and CI need no key;
  a live provider only when an environment variable enables it), an extractive fallback that answers with the best
  sentences when no LLM is configured, a cross-encoder reranker, citation checking (every cited span must be one of the
  retrieved chunks), and the API.
- **Day 4, evaluation.** A labelled question set (synthetic, written against the pinned corpus, each question with its
  gold chunk spans), recall@k and MRR per retriever, faithfulness checks on answers, and a CI job that fails when
  recall@k drops below the committed baseline.
- **Day 5.** The ablation table (full-text vs vector vs hybrid vs hybrid + rerank), README results, WALKTHROUGH.md,
  v1.0.0.

## Testing

- **Chunking:** example tests for each rule and property tests (Hypothesis) that generate random Markdown (headings,
  paragraphs, lists, fenced code with `#` lines and blank lines inside, very long lines and words) and check C1 to C7.
- **Normalization:** one test per rule, idempotence, and a golden file for a page that uses every extension.
- **Fetch:** tests build small tar archives in a temporary directory (with and without the pax commit header, with a
  tampered file, with a missing file, with a path that escapes the target) and run `fetch` on them; no network.
- **Database:** migrations, ingest and its idempotence against a real PostgreSQL (the local cluster, or CI's service
  container); each test gets its own schema, dropped afterwards.
- **CI:** lint (ruff), types (mypy), tests with the pgvector service container, and a corpus job that fetches the real
  pinned archive, checks it against the manifest, and chunks the whole corpus with the invariants checked.
