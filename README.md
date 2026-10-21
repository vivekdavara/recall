# Recall

Retrieval-augmented question answering over a documentation set, with a citation for every claim and an evaluation
harness that measures retrieval and answer faithfulness. PostgreSQL 17 does all the retrieval: pgvector (HNSW) for
meaning, full-text search for exact words, fused by reciprocal rank fusion, then reranked by a small cross-encoder.
Everything runs on a laptop CPU with no API key.

[DESIGN.md](DESIGN.md) is the specification. [WALKTHROUGH.md](WALKTHROUGH.md) answers the questions an interviewer is
likely to ask.

## Status

| Day | What | State |
|---|---|---|
| 1 | Design; the corpus (fetched, checked, normalized); chunking; the schema and ingest; the local database; CI | done |
| 2 | Local embeddings in pgvector (HNSW), full-text retrieval, hybrid by reciprocal rank fusion | planned |
| 3 | Answers with citations (recorded LLM provider, extractive fallback), a cross-encoder reranker, the API | planned |
| 4 | The eval harness: labelled questions with gold passages, recall@k, MRR, faithfulness; a CI gate | planned |
| 5 | The ablation table, results, v1.0.0 | planned |

## Why it is built this way

- **One database.** Documents, chunks, vectors and full-text indexes live in PostgreSQL, so ingest is one transaction
  and hybrid retrieval (day 2) is one SQL query rather than a join across a vector store and a search engine.
- **Citations point at exact spans.** A chunk is a verbatim slice of its normalized page (`content[start:end]`), so a
  citation can show exactly what was retrieved and link to the section it came from.
- **The corpus is reproducible.** The archive's bytes aren't pinned (GitHub regenerates archives); the commit id the
  archive declares and the SHA-256 of each of the 421 files used are, so anyone gets the same corpus or an error.
- **Measured, not asserted.** Day 4's harness turns "hybrid retrieval is better" into recall@k and MRR per retriever,
  and CI fails when they drop.

## Architecture

```mermaid
flowchart LR
    A[GitHub archive<br/>pinned commit] -->|recall fetch:<br/>commit id + manifest| B[data/corpus<br/>421 files]
    B -->|normalize| C[125 pages of<br/>plain Markdown]
    C -->|chunk| D[1,503 chunks<br/>with spans]
    D -->|recall ingest| E[(PostgreSQL 17<br/>documents, chunks)]
    E -.->|day 2| F[pgvector HNSW +<br/>full-text GIN]
    F -.->|days 2-3| G[hybrid retrieval, rerank,<br/>answer with citations]
    G -.->|day 4| H[eval harness<br/>recall@k, MRR]
```

## Layout

```
src/recall/
  markdown.py             lines, fenced code blocks and ATX headings (the parts both stages need)
  chunking.py             sections, blocks, greedy packing by tokens; invariants C1-C8
  corpus/source.py        the pinned source and which pages are kept
  corpus/fetch.py         download, the commit and manifest checks, all-or-nothing extraction
  corpus/manifest.json    the 421 files used, with sizes and SHA-256
  corpus/normalize.py     FastAPI's Markdown extensions to plain Markdown
  corpus/load.py          an extracted corpus to Documents
  ingest.py               documents and chunks into PostgreSQL, idempotent, one transaction
  db.py                   connections and migrations
  migrations/             numbered SQL files
  cli.py                  the recall command
scripts/pg.sh             the project-local PostgreSQL 17 cluster (port 5544)
tests/                    unit, property (Hypothesis), database and real-corpus tests
```

## How to run

Needs Python 3.11 and PostgreSQL 17 with pgvector (`brew install postgresql@17 pgvector` on macOS).

```bash
make venv                      # .venv with recall and the dev tools
scripts/pg.sh start            # the local cluster on port 5544: databases recall and recall_test
.venv/bin/recall fetch         # FastAPI 0.157.0's docs into data/corpus (about 18 MB downloaded, 421 files kept)
.venv/bin/recall verify        # check data/corpus against the manifest
.venv/bin/recall stats         # chunk the corpus and print its sizes
.venv/bin/recall show tutorial/query-params.md   # one page's chunks, with spans and heading paths
.venv/bin/recall ingest        # load it into the recall database (runs the migrations first)
make check                     # ruff, mypy --strict, and the tests (they use recall_test)
scripts/pg.sh stop
```

`recall ingest --max-tokens N` and `recall stats --max-tokens N` chunk with another size; ingest then re-chunks every
document. Settings: `RECALL_DATABASE_URL` (default `postgresql://localhost:5544/recall`), `RECALL_TEST_DATABASE_URL`
(default `.../recall_test`), `RECALL_DATA_DIR` (default `data`).

## Results (day 1)

The corpus, chunked with the default `max_tokens` 256 (`.venv/bin/recall stats` at `0e87c41`, chunker version 2;
the numbers are deterministic, and CI's corpus job prints the same table from a fresh download):

| | |
|---|---|
| Pages kept | 125 of FastAPI's 156 English pages |
| Files in the manifest | 421 (125 pages, 295 included code files, the license) |
| Normalized text | 972,537 characters |
| Chunks | 1,503 |
| Tokens (Recall's measure) | 217,390 |
| Tokens a chunk | min 7, p50 147, p95 250, max 256; 20 chunks at the maximum |

Before the last cut of each section was evened out (chunker version 1), the same 1,503 chunks had min 3, p50 142,
p95 253, 27 at the maximum, and three chunks that were only a closing fence; chunks under 20 tokens went from 41 to
30.

The same commit's archive by tag and by commit id differ in bytes (18,018,516 and 18,023,996, since the top
directory is named after what was asked for) and give the same manifest, which is why the files are pinned rather than
the archive.

`recall ingest` at `ba23111`: 125 added and 1,505 chunks written, then 125 unchanged and nothing written. After the
front-matter fix: 2 updated (the two pages that have front matter), 123 unchanged, 49 chunks written, 1,503 stored;
then 125 unchanged. After chunker version 2 (and the one-pass normalizer, which dropped `index.md`'s CSS): 125
updated, 1,503 chunks written, then 125 unchanged. Every stored chunk equals `substr(document.content, start + 1, end - start)` (checked in SQL, and by
`test_ingest.py`).

## Tests

`make check` runs ruff, `mypy --strict` and pytest: 124 tests at `4dfd206`, all passing here. In CI's test job the 7
real-corpus tests skip (116 passed, 7 skipped at `40a7144`, line and branch coverage 98%), and its corpus job runs them.

- **Chunking:** one example per rule, and Hypothesis properties for C1 to C8 over random documents built from known
  parts (headings, paragraphs, lists, fences with `#` lines, blank lines and non-closing fence runs inside), so the
  test knows where each heading and fence is; and the rules that need no structure over arbitrary text.
- **Normalization:** one test per rule, the error cases, and a golden page that uses every extension.
- **Fetch:** archives built in the test (another commit, a changed file, a missing file, a symlink, duplicate members,
  a member outside the top directory, unsafe manifest paths, an archive over the size limit); no network.
- **Database:** migrations (once, in order, concurrent migrators); ingest (idempotence, changed, retitled and removed
  documents, new settings, a failure part way rolls back, concurrent ingests queue), each test in its own schema.
- **The real corpus** (`tests/test_real_corpus.py`): every page loads, normalizes idempotently, has no build syntax
  left, and chunks within the invariants at two sizes; the manifest holds exactly what the pages use; a pin of the
  chunk count for this chunker version. Skipped without a fetched corpus, required in CI's corpus job.

- **Normalization properties:** generated pages keep their own code unchanged, get one block per include, keep every
  visible word, have no build syntax, tags or comments left outside code, and normalize idempotently.
- **The local cluster** (`tests/test_pg_script.py`): `scripts/pg.sh` start, status, url, psql, stop, a restart that
  keeps the data, and reset (which leaves a directory that isn't a cluster alone), on a throwaway cluster.

CI (GitHub Actions) has three jobs: lint, types and tests against a `pgvector/pgvector:pg17` service container; the
corpus job, which downloads the pinned archive from GitHub, verifies it and runs the real-corpus tests; and the
pg-script job, which installs PostgreSQL 17 and pgvector on Ubuntu and runs the cluster script's test.

## Review of day 1

Two review agents, one after the other, each at a fixed commit in its own worktree, worded as correctness reviews,
saving every bug they reproduced as a failing test (kept in `tests/review/` and `tests/review2/`); each was reproduced
here before it was fixed.

- **Review 1** (at `40a7144`, 15 minutes): the chunker held up under 20,000 fuzzed inputs; 23 failing tests elsewhere.
  Normalization handled fences, comments and code spans in separate passes that disagreed: a comment around a code
  block kept the code, `<!--` inside a code span swallowed the page up to the next `-->`, a raw include inside a
  comment was expanded, `index.md`'s `<style>` CSS was page text, and the manifest missed includes on CRLF pages. Now
  one pass in document order, and `includes()` is the set of files `normalize()` reads. Also: a failed rename lost
  the corpus; `verify` followed a symlinked directory; a BOM hid a page's title; `# #` had the title `#`; concurrent
  migrators and ingests failed under a REPEATABLE READ default (READ COMMITTED is now set before each advisory lock);
  bad downloads and a missing database printed tracebacks. 22 of its 37 mutations survived. Two were dead code (now
  removed) and one is `load`'s "only files in the manifest" check, which `verify` makes unreachable; for the rest I
  added tests and ran each mutant again (with the fixes' own two mechanisms, 23 mutants): 21 caught at first, and the
  two that survived (deleting a page exclusion) showed my exclusion test iterated over the list it was testing. It
  has its own list now, and both are caught.
- **Review 2** (at `0e87c41`): on the fixes and everything else since review 1.

One judgement call went against the review: it expected a raw include outside a fence to become a fenced block, as
DESIGN.md said; the documentation build inserts it as Markdown, so Recall now does too and DESIGN.md says so.

## Design decisions

- **FastAPI's docs as the corpus.** MIT-licensed, one subject, many near-duplicate passages that differ in detail, and
  questions that mix plain words with exact identifiers: a corpus where hybrid retrieval should matter.
- **Normalize before chunking, and keep offsets into the normalized text.** Code includes are expanded, so a question
  about code finds the code; spans then refer to what was embedded, not to source the reader never sees.
- **Chunks of whole blocks, no overlap.** Cuts fall between paragraphs and code blocks; each chunk carries its page
  title and heading path for context instead of repeating its neighbour's text.
- **A token measure of word runs and single punctuation.** Closer to word pieces than words, and exactly additive, so
  the size bound is exact. Day 2 checks it against the embedding model's own tokenizer.
- **One transaction for the whole ingest.** 1,503 chunks take well under a second, and readers never see a mix of
  two corpora; the cost is a long transaction for a much bigger corpus.
- **The builtin C.UTF-8 collation** locally and in CI, so text sorts by code point in both places.

## Limitations

- The Markdown scanner is not CommonMark: fences may be indented any amount (FastAPI indents fences in lists),
  setext headings are not headings, and HTML-like text outside code spans (`List<int>`) is stripped as a tag.
- Normalization is written for FastAPI's documentation build; another corpus needs its own rules.
- The token measure approximates the embedding model's tokens; a code-heavy chunk can have more model tokens.
- The local cluster uses trust authentication on localhost; it is for development only.
- One `recall fetch` at a time: two at once can leave one's set-aside copy behind (the corpus itself stays whole).
- An include line inside an included file is left as text (one level, as the build does), so normalizing such a
  page twice would expand it the second time; no page in the corpus has one.

## License

MIT (see [LICENSE](LICENSE)). The corpus is FastAPI's documentation (MIT, © Sebastián Ramírez); `recall fetch`
downloads it and keeps FastAPI's LICENSE beside it in `data/corpus/`; it is not committed here.
