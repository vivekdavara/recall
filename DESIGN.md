# Recall: design

Recall answers questions about a documentation set and cites the passages it used. It retrieves passages with
PostgreSQL 17 alone: pgvector (an HNSW index over local embeddings) for meaning, Postgres full-text search for exact
words, and reciprocal rank fusion to combine the two. A small cross-encoder reranks the fused list, and an answer is
written from the top passages with a citation for each claim. An evaluation harness measures retrieval (recall@k,
MRR) and answer faithfulness on labelled questions, and CI fails when retrieval gets worse.

This file is the specification. Day 1 built the corpus, its normalization and chunking, the storage schema and the
local database; day 2 the embeddings, full-text search, vector search and their fusion; day 3 the reranker, answers
with checked citations, and the API. The sections for later days say what they will add and are marked so.

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
left matches nothing. The query's size is bounded: each distinct phrase once, phrases of at most 16 lexemes, and
words in order while the query has at most 256 nodes (at most 64 words); ts_rank_cd's memory grows with the query,
and a 20,000-word question or one pasted list failed before (the day-2 reviews). Chunks matching the query (through the GIN index) are ranked by `ts_rank_cd` with the A/B/D
weights (1.0, 0.4, 0.1) and normalization 1 (divided by 1 + log of the chunk's length). This is not BM25: ts_rank_cd
counts how often and how close together the words occur but has no inverse document frequency, so a common word
("return", "response") counts as much as a rare one. Day 4's labelled set measured the cost: full-text finds a gold
chunk in its top 10 for 34% of the questions (vector 73%), and fusing its list in makes hybrid worse than vector alone
("Evaluation"). Day 5 measures BM25-style IDF weighting against it.

**Vector.** `ORDER BY embedding <=> question LIMIT n` through the HNSW index, with row numbers taken outside that
subquery so the index can still serve it (an `EXPLAIN` test checks). HNSW is approximate. An HNSW scan picks at most
`hnsw.ef_search` index entries (pgvector's default 40), so each query raises it to the list length for its own
transaction (before that, a list of 50 came back with 40); lists are at most 1,000 long, pgvector's largest
ef_search. Against an exact scan, 300 questions' top 10 had 0.9893 of the exact chunks at ef_search 40 and 0.9987 at
100 (`scripts/measure_retrieval.py`, each plan forced and checked). The floor stays at pgvector's 40: on 1,503 rows the
planner chooses by cost between the index and a sequential scan, and from about ef_search 80 to 100 it scans, so a
higher floor would stop using the index at all (a first measurement mistook that scan's exact results for HNSW's).
Hybrid's 50 candidates can already get a scan; both plans give the full list, the scan exactly. That alone wasn't enough: the scan picks its entries before dropping dead ones, and the index keeps the
entries of rows an ingest updated or deleted until VACUUM, so after a model change a list of 60 came back with 31
(found by the day-2 review). Each query also turns on pgvector's iterative scan (`hnsw.iterative_scan =
strict_order`), which keeps searching until the LIMIT is met, in distance order.

**Hybrid.** Both retrievers' top `candidates` (default 50) as two CTEs of one query, fused by reciprocal rank fusion
(Cormack, Clarke and Buettcher, 2009): score = the sum, over the lists a chunk is in, of 1 / (60 + rank). One query
means one snapshot and one round trip. RRF uses ranks, not scores, so it needs no calibration between ts_rank_cd and
cosine similarity; its known cost is that a chunk both lists rank moderately beats one that only one list ranks high
(the smoke test's CORS question: frontend.md at full-text 6 and vector 2 scores 0.03128, cors.md at 4 and 4 scores
0.03125). Ties go to the lower chunk id, so results repeat. `candidates` bounds how deep in one list a chunk can be and
still be fused in.

`recall search QUESTION --mode hybrid|full-text|vector -k 10` prints the hits; full-text search loads no model.

## Reranking

A cross-encoder reads the question and a passage together and scores how well the passage answers it. The
retrievers can't: each embeds or matches the question and the passages apart. Reading every pair is one pass of
the model each, too slow for the whole corpus, so it reranks only the top of the fused list: `ask` retrieves
hybrid's top 20 (`DEFAULT_DEPTH`) and answers from the reranked top 5. A chunk is scored with its heading path above
it, as it was embedded (`passage_text`).

**The model.** `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0), a 6-layer MiniLM trained on MS MARCO passage
ranking, pinned to revision `233902d2...` (a 91 MB weights file). It reads question and passage as one sequence of at
most 512 tokens (`[CLS] question [SEP] passage [SEP]`); a longer pair loses the passage's end. It has the same
WordPiece tokenizer as the embedder, and the same check applies (`scripts/token_fit.py --reranker --question ...`):
with a question attached, 1 to 3 of the 1,503 passages go over 512 (0.2% at most, for a 24-word question). Its
scores are logits, comparable only within one question.

**The interface.** `Reranker` has a `name` and `score(question, texts)`, one score per text, higher is better.
`rerank` orders hits by score, equal scores keeping the retrieval order, and refuses a wrong number of scores or one
that isn't finite. `CrossEncoderReranker` is the model (the `embed` extra); `OverlapReranker`, for tests, scores the
share of the question's distinct words in the text, so tests predict its order exactly.

## Answers

`recall.answer.answer` writes an answer from the passages, and every claim that reaches the caller cites a span of
one of them. Two writers:

**The LLM**, through a `Provider` (`recall.llm`): one `Request` (system prompt, user prompt, a JSON schema) in, one
`Completion` (text, model, token counts) out.

- `RecordedProvider`, the default (`RECALL_LLM=recorded`), replays a recording: a JSON file in `recordings/` named
  after the SHA-256 of the request's three parts (canonical JSON), holding the request and the response. A recording
  whose stored request no longer hashes to its name is refused. A question whose exact prompt wasn't recorded gets
  the extractive answer. So tests and CI never need a key or the network.
- `AnthropicProvider`, only when `RECALL_LLM=anthropic`: Claude (`claude-opus-5-5` unless `RECALL_LLM_MODEL` says
  otherwise) through the official SDK, which finds the key in the environment; structured output
  (`output_config.format` with the answer's JSON schema) so the response is JSON of the right shape, effort
  `medium`, and the server-side fallback (`fallbacks: "default"`), which re-runs a request the model's safety
  classifiers decline on the model recommended for that kind of refusal. A refusal, an answer cut off at
  `max_tokens`, or any error the SDK's own retries didn't fix is a `ProviderError`. With `RECALL_RECORD=1` each live
  completion is saved as a recording. It is tested against a local stand-in for the API, not against Claude: there
  is no key here.
- `RECALL_LLM=none`: always the extractive answer.

The prompt (`build_request`) numbers the passages from 1, each with its page and heading path, then the question;
the system prompt says to answer only from them, in short claims, each citing passages by number with a quote
copied word for word, and to say when they don't answer. It is deterministic, so a recording of it is found again.

**Citations** (`recall.citations.check`). The answer must be `{"answerable": bool, "claims": [{"text", "citations":
[{"passage", "quote"}]}]}`. A citation counts only if its passage is one of those given (an integer from 1, not a
boolean or a float) and its quote is in that passage's text: exactly, or with any run of whitespace standing for any
other (models re-wrap what they copy), and a quote that starts or ends with a letter, digit or underscore must start or
end a word there ("secure by default" is not in "insecure by default"). A quote must hold at least 8 characters besides whitespace, since one word
would match almost any passage. A citation that counts carries the chunk, the page and the span: offsets into the
page's normalized text, with `document[start:end]` the passage's own text that the quote matched. A claim with no
citation that counts is dropped; at most 20 claims and 5 citations a claim are checked. Each drop is reported, with
its reason, in the answer's `problems`. Only a response that isn't an answer at all (not JSON, or not that shape at
the top) raises `MalformedAnswer`. The check is about grounding, not truth: a quote shows where a claim came from,
not that the claim follows from it. Day 4 measures that (faithfulness).

**The extractive answer** (`recall.extractive`), when there is no provider, the provider fails, its response isn't an
answer, or none of its claims' citations check out: the passages' own sentences that the reranker scores highest for
the question, the best 3, best first, each a claim citing its own span. A sentence is prose: outside fenced code, not a
heading, a list item's marker left out; it ends at `.`, `!` or `?` followed by whitespace (and any closing quote,
bracket, `*` or backtick), except after a common abbreviation (`e.g.`). Sentences of fewer than 4 words ("Tip:") are
left out. A code block too long for one chunk is split across chunks, so a chunk can start inside one; read alone, its
code would be prose and its closing fence would hide the prose after it. So `ask` reads each passage's page
(`load_documents`) and sentences are found in the page; a page that no longer holds the passage (an ingest in
between) is not used. On the real corpus that changes 129 of 1,503 chunks' sentences (5,836 read alone, 5,496 in
their pages). The extractive answer can only say what the passages say, in their order of relevance, not as prose.

If the model says the passages don't answer the question, there is no answer (`method: none`), not an extractive one:
its sentences would look like an answer.

## The API

`recall serve` (uvicorn, `127.0.0.1:8109`) runs `recall.api.create_app()`:

- `POST /ask` `{question, mode, passages, depth, candidates, rerank}`: the answer as text (each claim followed by its
  passage numbers), its claims with their citations (passage, chunk, page, URL, quote, span), the passages, what
  answered (`method`, `provider`, `model`), `problems`, and each step's time in milliseconds.
- `POST /search` `{question, mode, k, candidates, rerank}`: the hits, reranked on request.
- `GET /healthz`: the corpus's size, the stored vectors' model, and the models in use; 503 if the database is
  unavailable or the stored vectors aren't the embedder's.

The models load once, at startup (`services_from_environment`: `RECALL_EMBEDDER`, `RECALL_RERANKER`, `RECALL_LLM`). The
handlers are plain functions, which FastAPI runs on its thread pool; each takes a connection from a pool (1 to 4
autocommit connections, a 5 s wait) for its database work only: the question is embedded before, and the reranker and
the LLM run after the connection is returned (`answer.retrieve`, then `answer.respond`). Holding it through the model
calls let four slow answers use up the pool, so /search and /healthz failed meanwhile (the day-3 review); a test runs
the app with a one-connection pool and takes that connection during each model call. Model calls run one at a time, under one lock:
PyTorch already spreads one call over the cores, so concurrent calls would mostly take turns, and Hugging Face's fast
tokenizers have failed under concurrent use in the past ("Already borrowed"). It is a precaution: 320 concurrent calls
from 8 threads without the lock raised nothing here (tokenizers 0.23.3).

Requests are validated: a question of 1 to 2,000 characters that isn't blank and is valid Unicode (JSON can carry a
lone surrogate, `"\ud800"`, which PostgreSQL can't take), sizes within bounds, `passages` at most `depth` when
reranking, no unknown fields. A bad request is a 422 that says where and why but not the input: FastAPI's own handler
echoes it back, and a lone surrogate can't be encoded, which made the 422 a 500. No database, no corpus, an older
schema, or vectors from another model are 503s, and the message says nothing of the URL or the server's error, either
of which can hold part of a password.

## Evaluation

`recall eval` measures retrieval on a labelled question set; `recall.evaluation` and `recall.faithfulness` are the
code.

**The questions** (`eval/questions.jsonl`, 110 of them, 55 of each kind; each question's first gold item is on one
of 66 pages, and the 12 questions with a second answering page bring the gold pages to 69). One JSON object a line: `id`,
`question`, `kind` and `gold`, a list of `{page, quote}`. They are **synthetic**: written by a Claude agent from the
pinned pages, read with `recall show`, with no access to the retrievers, their code or their results, so no question
was chosen or worded by what the system finds. Each `kind` is one of two: `identifier` (the question names an identifier or exact term from its answer, the
case full-text search is for) and `paraphrase` (the question describes the need in other words and holds none of the
answer's distinctive terms, the case vector search is for). A gold quote is the text that states the answer, copied
from the page's normalized text and found there exactly once; a question has more than one gold item only when
another page also answers it on its own.

**Gold chunks.** The labels name text, not chunk ids: a question's gold chunks are the chunks of a gold page whose span
overlaps a gold quote, found in the database each time the evaluation runs (`gold_chunks`). So re-chunking or
re-ingesting doesn't invalidate the labels; a quote that is no longer in its page exactly once, or lies only in a
heading line (headings aren't chunk text), fails the run with every such problem listed.

**Systems.** Each ranks the top 20 for every question: `full-text`, `vector`, `hybrid` (RRF of the two),
`vector+rerank` and `hybrid+rerank` (the vector or hybrid top 20 reordered by the cross-encoder; hybrid+rerank is
exactly what `ask` reranks by default). 20 is the reranker's depth, so recall@20 is the same with and without the
reranker by construction. Each question is embedded once for all systems.

**Metrics.** **recall@k**: the share of questions with a gold chunk in the top k (gold items are alternatives, so one
is enough), for k = 1, 3, 5, 10, 20. **MRR@20**: the mean over questions of 1 / the rank of the first gold chunk, 0
when none is in the top 20. Both overall and per kind. The report (`--json`) holds the settings, the scores and every
question's rank in every system, so a change can be traced to the questions it moved.

**Faithfulness** (`--answers`). Every question is answered as `recall ask` answers it (hybrid, reranked, 5 passages, the
provider `RECALL_LLM` names), and each claim is checked against the passages it cites: at least 75% of its distinct
content words (stop words dropped, a plural `s` stripped) must be in them, and every exact term must be found as
written, as a whole word (code spans; numbers, "1,500" as one; words with an underscore, a dot or a dash inside;
words with a capital after the first letter, so CamelCase and acronyms like JSON; not abbreviations like "e.g.").
Acronym plurals ("APIs" against "API") and hyphenated English ("built-in") are terms too, so a faithful paraphrase
can fail on them. That catches a claim that
changes a default or invents a parameter; it is lexical, so a claim that negates its passage in the passage's own
words passes (a test pins this limit). Per answer it also records whether a citation is a gold chunk and whether any
of its passages is. The answers are reported, not gated. With no recordings and no key here, every answer is
extractive, which is supported by
construction (its claims are the passages' sentences), so on this machine the faithfulness numbers test the harness;
the gold-citation shares are the informative ones. With `RECALL_LLM=anthropic RECALL_RECORD=1` the same run records
model answers, which CI would then replay.

**The gate.** `eval/baseline.json` is a report committed with the code. CI's model job runs `recall eval --check
eval/baseline.json` on the real corpus with the real models and fails if any system's recall@k drops by more than one
question's worth or its MRR by more than 0.01 (slack for float differences between machines moving a near-tie), or if
the settings differ: the question file's SHA-256, the corpus commit, the chunker and chunk count, the models, depth,
candidates and the RRF k. A change that improves retrieval should commit a new baseline with it, so the gate then
holds the new level.

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
    subgraph online["Answer a question (day 2: retrieval; day 3: rerank, answer, API)"]
        Q[question] --> V[vector top-k]
        Q --> K[full-text top-k]
        F --> V
        G --> K
        V --> R[reciprocal rank fusion]
        K --> R
        R -->|top 20| X[cross-encoder rerank]
        X -->|top 5| L{LLM provider:<br/>recorded by default}
        L -->|claims| C[citation check:<br/>quote in its passage]
        C -->|none pass, or no model| XT[extractive: best<br/>sentences, own spans]
        C --> API[FastAPI /ask,<br/>recall ask]
        XT --> API
    end
    subgraph eval["Measure (days 4-5)"]
        S[labelled questions<br/>with gold passages] --> M[recall@k, MRR,<br/>faithfulness]
        M --> CI[CI gate]
    end
```

## Later days (plan)

- **Day 2, retrieval.** Done: "Embeddings" and "Retrieval" above.
- **Day 3, answers.** Done: "Reranking", "Answers" and "The API" above.
- **Day 4, evaluation.** Done: "Evaluation" above. The choice of `candidates` and the RRF k was measured
  (`scripts/sweep_fusion.py`, on two halves of the questions): no setting beats vector alone and a smaller k helps
  hybrid on both halves, so the fault is the full-text list and the defaults stay until it is fixed.
- **Day 5.** BM25-style weighting (inverse document frequency) for full-text search, measured against ts_rank_cd;
  then the default retrieval mode chosen from the ablation table (full-text vs vector vs hybrid, with and without the
  reranker), README results, WALKTHROUGH.md, v1.0.0.

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
- **Answers** (day 3): the provider interface (request keys, recordings replayed and refused when edited, the
  recorder, the live provider's request and failures with a fake client, and the real SDK against a local stand-in
  for the API); the citation check (one test per rule and reason, and Hypothesis: no JSON value makes it raise
  anything but MalformedAnswer, and any quote taken from a passage checks out with the span it came from); sentence
  splitting and the extractive answer (each rule, a chunk inside a split code block, Hypothesis over generated
  Markdown); the reranker (order, ties, bad scores, the stub; the real model's rankings, pinned name and length);
  `answer` and `ask` with every fallback; the API with a test client (answers, a replayed recording, search, health,
  every 422 and 503); `recall ask` and `recall serve`.
- **CI**, four jobs: lint (ruff), types (mypy, strict) and the tests with the pgvector service container; a corpus
  job that fetches the real pinned archive from GitHub, checks it against the manifest and runs the real-corpus tests;
  a pg-script job that installs PostgreSQL 17 and pgvector on Ubuntu and runs the cluster script's test; and a model
  job (CPU-only PyTorch, the model cached between runs) that runs the `model` tests on the real corpus and writes the
  token fit and the smoke questions' ranks to the run's summary. The last three set an environment variable that turns
  their tests' skips into failures.
