# Recall: design

Recall answers questions about a documentation set and cites the passages it used. It retrieves passages with
PostgreSQL 17 alone: pgvector (an HNSW index over local embeddings) for meaning, Postgres full-text search for exact
words, and reciprocal rank fusion to combine the two. A small cross-encoder reranks the fused list, and an answer is
written from the top passages with a citation for each claim. An evaluation harness measures retrieval (recall@k,
MRR) and answer faithfulness on labelled questions, and CI fails when retrieval gets worse.

This file is the specification. Day 1 built the corpus, its normalization and chunking, the storage schema and the
local database; day 2 the embeddings, full-text search, vector search and their fusion. The sections for later days say
what they will add and are marked so.

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
MIT license, at tag `0.157.0`, commit `386a28af2f87af1a32ac26d88f07b7f58ab2dfb1`. It is a good test of retrieval: 125
pages on one subject (of the 156 English pages; the rest are excluded below), so many passages are near-duplicates in
topic and differ in detail (path parameters vs query parameters vs body parameters), and its questions mix plain words ("what status code does a validation error
get?") with exact identifiers (`Depends`, `HTTPException`, `response_model_exclude_unset`) that dense retrieval alone
tends to miss.

**Not committed.** `recall fetch` downloads the archive of the pinned commit and extracts the pages it uses, the code
files they include (`docs_src/`), and FastAPI's `LICENSE`, into `data/corpus/` (gitignored). The MIT license allows
redistribution, but fetching keeps the repository small and makes the source of every byte explicit.

**Integrity.** The archive's own bytes are not pinned: GitHub generates archives on request and has changed their
compression before, which changes the archive's hash without changing its contents. (The archives of the same commit by
tag and by commit id already differ: 18,018,516 and 18,023,996 bytes, because the top directory is named after what was
asked for; both give the same manifest.) Instead `fetch` checks two things:

1. The commit id the archive declares (git writes it into the tar's global pax header, `comment=<commit>`) must be
   the pinned commit.
2. Every file it extracts must match `src/recall/corpus/manifest.json`: the path, the size and the SHA-256 of each
   of the 421 files Recall uses (125 pages, the 295 files they include, and the license). A file the manifest lists but the archive lacks, or a file whose hash differs, fails the fetch, and
   nothing is written unless every file checks out (extraction goes to a temporary directory that is renamed into
   place at the end). Only regular files are read from the archive, by the manifest's paths, so a link or a member
   named `../x` in the archive can't write outside `data/corpus/`; two members for one path are refused.

`recall verify` checks an extracted corpus against the manifest (missing, changed and extra files), and `recall fetch`
does nothing when the corpus already matches it.

`recall manifest` regenerates the manifest from an archive; it is run once, when the pinned version changes, and the
diff is reviewed like code.

**Which pages.** The pages under `docs/en/docs/` except these, each for a reason:

| Excluded | Why |
|---|---|
| `release-notes.md` | 724 KB of changelog, half the corpus's bytes; its lines name versions, not behaviour, and would crowd out the pages that explain it. |
| `reference/**` | Generated API reference: each page is a `::: fastapi.X` directive that mkdocstrings expands from docstrings at build time; the Markdown holds almost no text. |
| `_llm-test.md`, `translation-banner.md` | Inputs to the translation tooling, not documentation. |
| `fastapi-people.md`, `external-links.md`, `management.md`, `newsletter.md` | Generated lists of people and links, project administration, a sign-up page. |

Every exclusion is a path rule in `recall.corpus.source`, tested, and the manifest lists exactly the pages kept. One
page includes FastAPI's own source (`fastapi/openapi/docs.py`, for the Swagger UI defaults); the manifest lists it
like any other included file.

## Normalization

FastAPI's Markdown uses extensions of its documentation build. `recall.corpus.normalize` turns each page into plain
Markdown that reads the way the rendered page does. The normalized text is the document Recall stores, and every
offset Recall reports (chunk spans, citations) is into it.

First the page is read the way the build reads it: a UTF-8 byte order mark goes, CRLF becomes LF, a NUL is an error,
and **front matter** goes (a first line `---` up to the next `---` or `...` line: page settings such as `index.md`'s
sponsors file and `advanced/templates.md`'s `render_macros: false`; a `---` anywhere else is a horizontal rule and
stays).

Then one pass over the lines, in order, in one of three states. Whichever starts first wins, as in CommonMark: a fence
inside an HTML comment is part of the comment, and a comment inside a fence is code.

- **In a fenced code block:** the line is code and is kept as it is. The one exception is a raw include
  (`{!path!}`, `{!> path!}`), which FastAPI's pages put inside fences to fill them: the included file's lines become
  the fence's lines. A file with a line of only backticks or tildes, which could close the fence, is refused.
- **In an HTML comment, or a `<style>` or `<script>` element:** the line becomes a blank line, up to the comment's
  `-->` or the element's end tag, whatever its attributes hold (`<style-guide>` is another element); what follows on
  that line is page text. (`index.md` has a `<style>` block whose CSS the rendered
  page never shows.)
- **In page text:**
  1. A fence line opens a fenced block.
  2. A **code include** (`{* path ln[a:b,c] hl[...] title[...] *}`, a line of its own) becomes a fenced block of the
     Python file, with the lines `ln` selects (1-based, inclusive ranges, in increasing order, no overlap; a line
     `...` stands for lines left out between ranges, as the rendered page marks them) and a fence longer than any
     backtick run in the code; `hl` and `title` only change how the block is drawn, so they go. A **raw include**
     outside a fence inserts the file's text, which is then read as the page's own text (the build's
     markdown-include inserts it as Markdown); include lines inside an included file are not expanded (one level).
  3. **Admonitions** (`/// tip`, `/// note | Technical Details`) become a label line (`Tip:`, `Note: Technical
     Details`) followed by their body, and **tabs** (`//// tab | Python 3.10+`) become `Python 3.10+:`; a closing
     `///` or `////` becomes a blank line, so blocks stay apart.
  4. In everything else, headings included, **HTML comments** and **tags** are removed and the text between tags is
     kept (`<dfn title="...">parsing</dfn>` becomes `parsing`), except inside **code spans**, which are found over
     the whole paragraph (included lines and the page's own form one paragraph, in either order), so a span may wrap
     onto the next
     line: `` `<b>` `` and `` `<!--` `` are code. A comment that opens and closes on an include or marker line doesn't
     hide it, and a marker's label loses its markup like any text. A text line that reads as block syntax only
     because markup went (`<b>#</b> Title` is `# Title`, `<!-- x -->```` is a fence opener) gets a backslash
     (CommonMark's escape: `\# Title`), since the rendered page shows that text, not a heading or a fence. Heading ids (`## Defaults { #defaults }`)
     stay in the heading line: the chunker reads them as the section's anchor and leaves them out of the title, so a
     citation can link to `https://fastapi.tiangolo.com/tutorial/query-params/#defaults`.

Includes are written relative to `docs/en/`, the build's root, and must resolve inside the repository; a missing one is
an error, not a silent gap. `includes(page)` is defined as the files `normalize` reads, so the manifest and the
normalizer can't disagree about what a page uses. Normalization is deterministic and idempotent: normalizing its own
output changes nothing (tested on every real page, a golden page and generated pages). An include line inside an
included file is not expanded (one level, as the build does); it is escaped like any text line that reads as block
syntax, so a second pass doesn't expand it either, and a file included inside a fence may not hold one at all (as
code it would stay unescaped). Output that would start with front matter (a rule after the real front matter, an
included file's first lines) gets a blank line first, so a second pass keeps it.

## Chunking

A chunk is the unit that is embedded, indexed, retrieved and cited. `recall.chunking` splits a normalized document in
three steps:

1. **Sections.** ATX headings (`#` to `######`, outside fenced code) split the document into sections. Each section
   has a heading path (the titles of the headings above it, outermost first) and an anchor. The heading lines
   themselves are not chunk text; the heading path carries them.
2. **Blocks.** A section's body splits at blank lines into blocks, except that a fenced code block is one block, blank
   lines and all.
3. **Packing.** Blocks are packed greedily, in order, into chunks of at most `max_tokens` tokens. A block bigger than
   `max_tokens` alone is split at line ends, a line bigger than that at spaces, and a run of non-whitespace bigger
   than that (a 300-character URL, say) between two tokens; the pieces are packed greedily the same way. A chunk is
   cut inside a run of non-whitespace only in that last case. Last, the cut between a section's last two chunks is
   moved to where the bigger of the two is smallest (on a tie, the later cut), so a section doesn't end in a remainder
   of a few tokens: greedy packing alone left three chunks that were only the closing fence of a long code block. The
   move can't change the number of chunks: the two still don't fit in one, and the first still starts where it did.

**Size measure.** A token is a run of word characters or one punctuation character (`\w+|[^\w\s]`). It is closer to
the word pieces an embedding model sees than counting words is (`@app.get("/items/{item_id}")` is 14 tokens, not 1),
and it never spans whitespace, so a chunk's count is exactly the sum of its blocks' counts and the bound can be checked
exactly. The default `max_tokens` is 256. In the embedding model's own tokens (WordPiece, with the heading path that is
embedded with the chunk) the corpus's 1,503 passages are p50 187, p95 339, p99 422 and max 723; the model reads 512, so
one passage is truncated: a base64 PNG in a code block, which is one token here and hundreds of word pieces there
(`scripts/token_fit.py`, "Embeddings" below). So `max_tokens` stays 256.

**No overlap.** Chunks do not overlap. Overlap is a common way to keep context that a cut separates; here cuts fall
at block boundaries, and each chunk carries its page title and heading path, which day 2 prepends to the text it
embeds. Overlap would also make the same sentence appear in two retrieved passages and two citations.

**Invariants** (each has a test, most of them a property test over random Markdown):

- **C1 Verbatim.** A chunk's text is exactly `document[start:end]`, so a citation's span shows what was retrieved,
  and it starts and ends with a non-whitespace character.
- **C2 Order.** Chunks are in document order and do not overlap: each starts at or after the previous one's end.
- **C3 Coverage.** Every non-whitespace character of the document is in exactly one chunk, except heading lines.
- **C4 Size.** Every chunk has at least 1 and at most `max_tokens` tokens.
- **C5 Sections.** A chunk lies inside one section, and its heading path and anchor are that section's.
- **C6 Code.** A fenced code block of at most `max_tokens` tokens is never split across chunks.
- **C7 Deterministic.** The same document and settings give the same chunks.
- **C8 Maximal.** Two consecutive chunks of one section never fit in one chunk together.

## Storage

PostgreSQL 17 with pgvector holds everything. One database for documents, chunks, vectors and full-text indexes means
one transactional ingest, and day 2's hybrid retrieval is one SQL query instead of a join across two systems.

Day 1's schema (`src/recall/migrations/001_documents_and_chunks.sql`):

- `documents`: `id`, `path` (unique, e.g. `tutorial/query-params.md`), `title`, `url`, `content` (the normalized
  text), `content_sha256`, `ingested_at`.
- `chunks`: `id`, `document_id` (cascade delete), `ordinal` (unique per document), `heading_path text[]`, `anchor`,
  `start_offset`, `end_offset`, `content`, `token_count`. Check constraints mirror C1 and C4 where SQL can see them
  within a row: `0 <= start_offset < end_offset`, `char_length(content) = end_offset - start_offset`, and
  `token_count >= 1`. That `content` is the right slice of its document (`substr(document.content, start + 1, end -
  start)`) needs the document's row, so the ingest tests check it in SQL rather than a trigger.
- `corpus_settings`: one row, what the stored chunks were made from and with (source commit, chunker version,
  `max_tokens`).
- `schema_migrations`: which migrations ran, so `recall migrate` applies each file once, in order, each in its own
  transaction, under an advisory lock so that concurrent migrators queue.

**Ingest** (`recall ingest`) makes the stored corpus exactly the loaded one, in one transaction, so a reader sees the
old corpus or the new one and never a mix. For each document: if a row with the same path, content hash, title and URL
exists and the chunks were made with the same settings, nothing changes; otherwise the document row is inserted or
updated and its chunks are replaced. Documents that are no longer in the corpus are deleted, with their chunks. Running
ingest twice leaves the same rows (the same chunk ids), and the second run reports every document unchanged. The
settings the chunks were made with are stored in `corpus_settings`: the source commit, `max_tokens`, and
`CHUNKER_VERSION` (raised whenever a change to the chunker can change a chunk); when any of them differs, every
document is re-chunked rather than mixing chunk sizes. Two ingests queue on an advisory lock, so they never interleave;
the second sees what the first committed. A failure part way (a row PostgreSQL refuses) rolls the whole ingest back.

Since day 2, ingest also embeds, in the same transaction, after writing the chunks: every chunk whose `embedding` is NULL
(the chunks it just wrote) is embedded, 64 per model call and per `UPDATE`. When the model named in `corpus_settings`
differs from the one ingesting, every stored vector is set to NULL first, so all are made again, while unchanged chunks
keep their rows and ids. A model that fails part way, returns the wrong number of vectors or the wrong dimension, or
returns a NaN (pgvector refuses it) rolls everything back, vectors included.

Day 2's migration (`002_retrieval.sql`) adds:

- `chunks.embedding vector(384)`, unit length, with an HNSW index for cosine distance (pgvector's defaults, `m = 16`,
  `ef_construction = 64`). It is NULL only inside an ingest, between writing a chunk and embedding it.
- `chunks.search_vector`, a generated `tsvector` with a GIN index: the page title (`heading_path[1]`, the page's first
  heading) weighted A, the headings under it B, the chunk text D, all with the `english` configuration (stemming, stop
  words). Generated, so it can't disagree with the row. `array_to_string` is STABLE (an element type's output can depend
  on settings) and a generated column may only call IMMUTABLE functions, so a small wrapper, `heading_text(text[])`,
  declares what is true for `text[]`; its SQL-standard body is bound when it is created, so the search path can't
  change it later.
- `corpus_settings.embedding_model`: the model and revision that made the stored vectors (the dimension is the column's
  type).

A day-1 database migrates in place: its chunks keep their ids, get their search vectors at once, and the next ingest
embeds them.

## Embeddings

**The model.** `BAAI/bge-small-en-v1.5` (MIT license), 384 dimensions, 33.4M parameters, a 133 MB weights file, run on the CPU through
sentence-transformers and pinned to a revision of its Hugging Face repository (`5c38ec7c...`), so a re-upload can't
change the vectors. The other common small model, `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, also 384
dimensions), has the same WordPiece tokenizer but reads only 256 tokens: on this corpus it would truncate 369 of 1,503
passages (24.6%), against bge-small's 1 (`scripts/token_fit.py`; "Chunking" above). (Its authors also report it the
stronger retriever on the public MTEB benchmark; not measured here, and day 4's evaluation is the test that counts.)

**What is embedded.** A chunk's heading path, page title first, joined by ` > `, a blank line, then the chunk text
(`passage_text`). A chunk alone often doesn't say what it is about ("As query parameters are not a fixed part of a
path...") and its heading line isn't chunk text, so the path supplies what the rendered page shows above it. A question
gets the prefix the model's authors give for search queries ("Represent this sentence for searching relevant passages:
"); passages get none. Vectors are scaled to unit length, so cosine distance is 1 minus the dot product.

**The interface.** `Embedder` has a `name` (the model and its revision: what `corpus_settings.embedding_model` holds), a
`dimension`, and separate `embed_passages` and `embed_query`. Two implementations:

- `SentenceTransformerEmbedder`, the real one: the optional `embed` extra (`pip install -e '.[embed]'`), loaded only
  when a command needs vectors. It checks the model's dimension against the column's.
- `HashEmbedder`, a stub with no dependencies for tests: each token (as the chunker counts them, lowercased) adds +1 or
  -1 to one of 384 buckets, both chosen by a hash; the sum is scaled to unit length. Texts sharing tokens get similar
  vectors, so tests can predict rankings; it knows nothing of meaning. A text with no tokens gets a fixed unit vector
  (cosine distance to a zero vector is undefined; pgvector returns NaN).

`recall ingest --embedder model|hash` (default `model`, or `$RECALL_EMBEDDER`) chooses one. Search refuses vectors made by
another model than the one embedding the question: they would be compared in different spaces.

## Retrieval

`recall.retrieval` has three retrievers, each one SQL query, and `search`, which picks one, embeds the question when
needed and checks the model. Each returns `Hit`s: the chunk, its page, title, URL with the section's anchor, heading
path, span, score, and its rank in each retriever's list.

**Full-text.** The question is split at whitespace; each word becomes `phraseto_tsquery('english', word)` and the words
are ORed. OR, because `plainto_tsquery` ANDs every word, so a chunk would need all of them, and questions are written
in more words than any passage shares. A phrase per word, because the parser splits identifiers:
`response_model_exclude_unset` is four lexemes, and ORing them ranked every chunk about response models above the one
about that parameter (it was 4th; as a phrase, 1st). Words that are only stop words drop out; a question with none
left matches nothing. Chunks matching the query (through the GIN index) are ranked by `ts_rank_cd` with the A/B/D
weights (1.0, 0.4, 0.1) and normalization 1 (divided by 1 + log of the chunk's length). This is not BM25: ts_rank_cd
counts how often and how close together the words occur but has no inverse document frequency, so a common word
("return", "response") counts as much as a rare one. The smoke questions show the cost (a question in common words
misses its page); day 4 measures BM25-style IDF weighting against it on the labelled set.

**Vector.** `ORDER BY embedding <=> question LIMIT n` through the HNSW index, with row numbers taken outside that
subquery so the index can still serve it (an `EXPLAIN` test checks). HNSW is approximate. An HNSW scan returns at most
`hnsw.ef_search` rows (pgvector's default 40), so each query raises it to the list length for its own transaction;
before that, a list of 50 came back with 40.

**Hybrid.** Both retrievers' top `candidates` (default 50) as two CTEs of one query, fused by reciprocal rank fusion
(Cormack, Clarke and Buettcher, 2009): score = the sum, over the lists a chunk is in, of 1 / (60 + rank). One query
means one snapshot and one round trip. RRF uses ranks, not scores, so it needs no calibration between ts_rank_cd and
cosine similarity; its known cost is that a chunk both lists rank moderately beats one that only one list ranks high
(the smoke test's CORS question: frontend.md at full-text 6 and vector 2 scores 0.03128, cors.md at 4 and 4 scores
0.03125). Ties go to the lower chunk id, so results repeat. `candidates` bounds how deep in one list a chunk can be and
still be fused in.

`recall search QUESTION --mode hybrid|full-text|vector -k 10` prints the hits; full-text search loads no model.

## The local database

`scripts/pg.sh` runs a project-local PostgreSQL 17 cluster on port 5544 with its data in `.pgdata/` (gitignored):
`start` (runs `initdb` the first time, then `pg_ctl start`; listens on localhost only, trust authentication, so it
must not be exposed), `stop`, `status` (exit 3 when stopped), `url` (prints the connection URL), `psql`, and `reset`
(stop and delete the data directory, only if it is one: `PG_VERSION` must be there). It creates the `recall` database
for the app and the `recall_test` database for the tests, and checks that the `vector` extension is available before
saying it started. `initdb` uses the builtin `C.UTF-8` locale, so text sorts by code point; the server gets a valid
`LC_ALL`, without which PostgreSQL 17 on macOS refuses to start ("postmaster became multithreaded during startup").
`PG_BIN`, `RECALL_PG_PORT` and `RECALL_PG_DATA` override the binaries, the port and the data directory, which is how
the script's own test runs a throwaway cluster. In CI the database for the tests is the `pgvector/pgvector:pg17`
service container, initialised with the same locale.

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
        E -->|embed: bge-small| F[(pgvector HNSW)]
        E -->|generated tsvector| G[(full-text GIN)]
    end
    subgraph online["Answer a question (day 2: retrieval; day 3: the rest)"]
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

- **Day 2, retrieval.** Done: "Embeddings" and "Retrieval" above.
- **Day 3, answers.** A provider interface for the LLM (a recorded provider by default, so tests and CI need no key;
  a live provider only when an environment variable enables it), an extractive fallback that answers with the best
  sentences when no LLM is configured, a cross-encoder reranker, citation checking (every cited span must be one of the
  retrieved chunks), and the API.
- **Day 4, evaluation.** A labelled question set (synthetic, written against the pinned corpus, each question with its
  gold chunk spans), recall@k and MRR per retriever, faithfulness checks on answers, and a CI job that fails when
  recall@k drops below the committed baseline. Also BM25-style weighting (inverse document frequency) for full-text
  search, measured against ts_rank_cd, and the choice of `candidates` and the RRF k.
- **Day 5.** The ablation table (full-text vs vector vs hybrid vs hybrid + rerank), README results, WALKTHROUGH.md,
  v1.0.0.

## Testing

- **Chunking:** example tests for each rule and property tests (Hypothesis) that generate random Markdown from known
  parts (headings, paragraphs, lists, fenced code with `#` lines, blank lines and non-closing fence runs inside, very
  long lines and runs) and check C1 to C8 against what the generator knows, not against the chunker's own parse.
- **Normalization:** one test per rule and error case; a golden file for a synthetic page that uses every extension;
  property tests over generated pages (the page's own code unchanged, one block per include, no build syntax, tags
  or comments left outside code, every visible word kept, idempotent).
- **Fetch:** tests build small tar archives in a temporary directory (another commit or none, a changed file, a
  missing file, a symlink, duplicate members, a member outside the top directory, unsafe manifest paths, an archive
  over the size limit) and run `fetch` on them; no network.
- **Database:** migrations, ingest and its idempotence against a real PostgreSQL (the local cluster, or CI's service
  container); each test gets its own schema, dropped afterwards.
- **The real corpus** (`tests/test_real_corpus.py`): every kept page loads, normalizes idempotently, has no build
  syntax left and chunks within the invariants at two sizes; the manifest holds exactly what the pages use; a pin of
  the corpus's chunk count for the current `CHUNKER_VERSION`. Skipped when no corpus has been fetched.
- **The local cluster** (`tests/test_pg_script.py`): `scripts/pg.sh` end to end on a throwaway cluster.
- **Embeddings and retrieval** (day 2): the hash embedder's properties; ingest's embedding step (vectors equal to a
  fresh embedding of each passage, only new chunks embedded, a new model re-embedding everything with chunk ids kept,
  failures part way rolling back, a day-1 database migrated in place); the generated search vector's lexemes and
  weights; each retriever on small corpora whose rankings are predictable; hybrid checked against a Python fusion of
  the two retrievers' own lists (Hypothesis over questions, k, candidates and the RRF k); questions holding tsquery
  syntax or nothing searchable; `EXPLAIN` showing both indexes usable.
- **The real model** (`-m model`): unit-length 384-wide vectors, the model ranking three obvious passages right, and
  `tests/test_real_retrieval.py`: the real corpus ingested with the real model and searched in every mode (a smoke
  test with six hand-written questions, not an evaluation). They skip without the `embed` extra, except in CI's model
  job.
- **CI**, four jobs: lint (ruff), types (mypy, strict) and the tests with the pgvector service container; a corpus
  job that fetches the real pinned archive from GitHub, checks it against the manifest and runs the real-corpus tests;
  a pg-script job that installs PostgreSQL 17 and pgvector on Ubuntu and runs the cluster script's test; and a model
  job (CPU-only PyTorch, the model cached between runs) that runs the `model` tests on the real corpus and writes the
  token fit and the smoke questions' ranks to the run's summary. The last three set an environment variable that turns
  their tests' skips into failures.
