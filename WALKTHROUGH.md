# Walkthrough

Questions an interviewer is likely to ask about Recall, with short answers grounded in the code. Day 1 covers the
corpus, normalization, chunking, storage and ingest; day 2 embeddings and retrieval; day 3 reranking, answers with
citations and the API; later days add their own sections.

## The project

**What is Recall?**
A question-answering system over FastAPI's documentation that cites the passages it used. Retrieval is PostgreSQL 17
alone: pgvector for meaning, full-text search for exact words, combined by reciprocal rank fusion and reranked by a
cross-encoder (days 2 and 3). An evaluation harness (day 4) measures recall@k, MRR and faithfulness, and CI fails if
retrieval gets worse.

**What did day 1 build?**
Everything before retrieval: fetching a pinned corpus and proving it is the right one (`corpus/fetch.py`), turning
FastAPI's Markdown into plain Markdown (`corpus/normalize.py`), chunking with checked invariants (`chunking.py`), the
schema and an idempotent ingest (`ingest.py`, `migrations/001_...sql`), a local PostgreSQL 17 cluster
(`scripts/pg.sh`), and CI with a pgvector service container plus a job that fetches the real corpus.

**What did day 2 build?**
Retrieval. Chunks are embedded with a small local model (`embedding.py`) inside ingest's transaction and stored in
pgvector with an HNSW index; a generated, weighted `tsvector` gives full-text search (`migrations/002_retrieval.sql`);
`retrieval.py` has full-text, vector and hybrid retrieval, the last fusing the other two by reciprocal rank fusion in
one SQL query; `recall search` runs them; a CI job runs the real model on the real corpus.

**What did day 3 build?**
Answers. Hybrid's top 20 are reranked by a small cross-encoder (`rerank.py`); the top 5 go to an LLM provider
(`llm.py`: recorded responses by default, Claude only when `RECALL_LLM=anthropic`), whose JSON answer is checked
claim by claim (`citations.py`: every quote must be in the passage it cites); if no model answers usably, the answer is
the passages' best sentences (`extractive.py`). `answer.py` ties it together, `api.py` serves it (`POST /ask`), and
`recall ask` / `recall serve` run it.

## The corpus

**Why FastAPI's docs?**
MIT-licensed, so it can be used and redistributed; one subject, so many passages are close in topic and differ in
detail (path vs query vs body parameters), which is where retrieval gets hard; and its questions mix plain language
with exact identifiers (`Depends`, `response_model_exclude_unset`), which is the case for hybrid retrieval.

**Why not commit the corpus?**
It keeps the repository small and makes the source of every byte explicit: `recall fetch` downloads one archive of a
pinned commit. The license would allow committing it.

**How do you know everyone gets the same corpus?**
Two checks in `fetch.extract`. The archive must declare the pinned commit (git writes the commit id into the tar's
global pax header, `comment=...`), and each of the 421 files used must have the size and SHA-256 in
`src/recall/corpus/manifest.json`. Extraction goes to a temporary directory that replaces `data/corpus` only if every
file checks out. CI's corpus job downloads it fresh on every push and gets the same 1,503 chunks.

**Why not just pin the archive's SHA-256?**
GitHub generates archives on request and has changed their compression before, so the bytes can change while the
contents don't. It is visible even now: the same commit's archive by tag and by commit id are 18,018,516 and 18,023,996
bytes (the top directory is named after what was asked for) and give the same manifest.

**What did you leave out, and why?**
`release-notes.md` (724 KB, half the corpus's bytes, about versions rather than behaviour), `reference/` (generated
from docstrings at build time; the Markdown is a directive), translation tooling inputs, and generated lists of people
and links. Each is a rule in `corpus/source.py` with its reason; 125 of 156 pages are kept.

**Could a malicious archive write outside `data/corpus`?**
No: `extract` never calls `tarfile.extract`. It reads members by the manifest's paths, only regular files (a symlink
or hard link where a file should be is refused), and writes them to paths built from the manifest, whose paths are
checked to be plain relative paths. Two members for one path, or a member outside the top directory, are refused too.

## Normalization

**What does it do, and why is it needed?**
FastAPI's pages use build extensions: `{* ../../docs_src/x.py ln[1:3] hl[2] *}` includes a code file, `/// tip`
starts an admonition, `//// tab | Python 3.10+` a tab, plus HTML. Normalization makes the page read the way the
rendered page does: includes become fenced code, markers become label lines, HTML goes. Without it, a question about
code would never match, because the code isn't in the Markdown.

**Why are offsets into the normalized text rather than the source file?**
Because that is what is chunked, embedded and shown. The normalized text is stored in `documents.content`, so a
citation's span can always be shown exactly.

**Why are raw includes expanded inside fences when nothing else is?**
FastAPI writes them inside fences (`` ```jinja `` then `{!../../docs_src/templates/item.html!}`): the include fills the
fence. An included file with a line that could close the fence is refused, so an include can't change the page's
structure.

**How do you test normalization beyond examples?**
Property tests (`tests/test_normalize_properties.py`) build pages from known parts (text with tags, code spans and
comments; code includes with options; raw includes in fences; the page's own fences holding `#`, `///`, `{*` and tag
lines; admonitions, tabs, multi-line comments) and check what each part must become: the page's own fences come out
unchanged and in order, each include adds one block, no build syntax, tag or comment survives outside code, every
visible word survives, and a second pass changes nothing. Processing lines inside fences, or leaving markers alone,
each fail it.

**Did normalization miss anything on the real pages?**
Front matter. Two pages (`index.md`, `advanced/templates.md`) start with YAML settings for the build between `---`
lines, and the first version kept them as text, so each became a chunk ("--- render_macros: false ---"). I found it
by listing the smallest chunks. Front matter is now removed, the real-corpus test checks no page starts with it, and
re-ingesting updated exactly those two documents.

**How do you know it handles every page?**
`tests/test_real_corpus.py`: every one of the 125 pages normalizes, normalizing the output again changes nothing, and
no include, admonition marker, HTML comment or tag is left outside code. A synthetic golden page covers every rule.

## Chunking

**How does chunking work?**
Three steps in `chunk_document`: headings split the page into sections; blank lines split a section into blocks (a
fenced code block is one block, blank lines and all); blocks are packed greedily into chunks of at most `max_tokens`
tokens. A block too big alone splits at line ends, a line at spaces, and a run with no spaces between two tokens.

**What is a token here, and why not use the model's tokenizer?**
A run of word characters or one punctuation character (`\w+|[^\w\s]`). It is closer to word pieces than counting
words, and no token contains whitespace, so a cut at whitespace or between tokens splits the count exactly: the bound is
exact without running a tokenizer. Day 2 measures how it compares with the embedding model's tokenizer.

**Why 256?**
A starting point that fits small embedding models' input (typically 256 to 512 word pieces) with room for the title
and heading path. It gives p50 147 and p95 250 tokens a chunk on this corpus (chunker version 2). Day 2 checks it against the real
tokenizer, and day 5's ablation can vary it.

**Why no overlap?**
Cuts fall between blocks, not mid-sentence, and each chunk carries its page title and heading path, so the context
overlap usually buys is already there. Overlap would also make the same sentence appear in two passages and two
citations.

**What are the invariants?**
C1 a chunk is exactly `document[start:end]`, starting and ending on text; C2 chunks are in order without overlap; C3
every non-whitespace character outside heading lines is in a chunk; C4 every chunk has 1 to `max_tokens` tokens; C5 a
chunk lies in one section with that section's heading path; C6 a fence that fits is never split; C7 deterministic;
C8 two consecutive chunks of one section never fit in one.

**How are they tested?**
Hypothesis builds random documents from known parts (headings, paragraphs, lists, fences containing `# comments`,
blank lines and runs of the fence character too short to close it), so the test knows where every heading and fence
is and checks each invariant against that, not against the chunker's own parse. The same checks run on every real
page at two sizes.

**Did greedy packing cause any problem?**
Three chunks on the real corpus were only "```": a code block too big for one chunk was split greedily at line ends,
the remainder was the closing fence, and the section ended there, so nothing absorbed it. Now the cut between a
section's last two chunks moves to where the bigger is smallest. It can't change the count (the two still don't fit
together, and the first still starts where it did), and it didn't: 1,503 chunks before and after, the smallest 3
tokens before and 7 after. My first attempt split big blocks evenly on their own, which stopped their first lines
filling the chunk before them and made 32 more chunks; I measured that and dropped it. C8 (two consecutive chunks of
a section never fit in one) is the property that keeps the argument honest.

**Why are heading lines not part of chunk text?**
They are carried by the heading path, which day 2 prepends to what is embedded; keeping them in the text would put
the same heading in every chunk of a long section, or in only the first.

**What happens to a 300-character URL?**
It is one run with no spaces; if it has more than `max_tokens` tokens it is cut between two tokens, the only case
where a chunk is cut inside a run of non-whitespace.

## Storage and ingest

**Why PostgreSQL rather than a vector database?**
One system holds documents, chunks, vectors and the full-text index, so ingest is one transaction and hybrid
retrieval is one query. There is one thing to run and back up; day 2 measures what pgvector's HNSW index costs at this
size rather than assuming it.

**How is ingest idempotent?**
Each document's content hash, title and URL are compared with the stored row; if they match and the chunk settings
(source commit, `max_tokens`, `CHUNKER_VERSION`) are the same, the document is skipped. A second run writes nothing and
keeps every chunk id (`test_ingesting_the_same_corpus_again_writes_nothing`).

**What is `CHUNKER_VERSION` for?**
A change to the chunker's code can change chunks without changing any document or setting. Raising the version makes
ingest re-chunk everything. `test_the_chunks_are_the_ones_this_chunker_version_makes` pins the corpus's chunk count,
so a change that alters chunks fails a test until the version is raised and the pin updated.

**Why one transaction for the whole ingest?**
Readers see the old corpus or the new one, never a mix, and a failure part way (a row PostgreSQL refuses) leaves the
previous corpus (`test_a_failure_part_way_leaves_the_previous_corpus`). Writing 1,503 chunks takes well under a
second; embedding them, since day 2, about 40 s on a laptop CPU, inside the same transaction. For a much bigger corpus
I would embed outside the transaction (vectors keyed by the passage text's hash) and load into a staging table and
swap, to keep the transaction short.

**What if two ingests run at once?**
They queue on an advisory lock taken at the start of the transaction; under READ COMMITTED, the second one's reads
after the lock see what the first committed, so it finds nothing to do (`test_concurrent_ingests_queue...`).

**Why the checks in the schema, like `char_length(content) = end_offset - start_offset`?**
They catch a broken writer at the database: an empty span, a negative offset, a chunk with no tokens, or content that
isn't as long as its span. Whether the content is the right slice needs the document's text, so the tests check that
in SQL with `substr`.

## The local database and CI

**Why the builtin C.UTF-8 locale?**
So text sorts by code point both on a Mac and in CI (where the container is initialised with the same arguments).
Collation differences between macOS and glibc have broken order-dependent tests in earlier projects.

**Why does `pg.sh` set `LC_ALL`?**
Without it, PostgreSQL 17's postmaster on this Mac refused to start ("postmaster became multithreaded during
startup"); the fix is a valid `LC_ALL` in the server's environment.

**What does CI run?**
Three jobs: ruff, `mypy --strict` and the tests against a `pgvector/pgvector:pg17` service container (trust
authentication, so there is no password); a corpus job that downloads the pinned archive from GitHub, verifies it,
runs the real-corpus tests and prints the corpus sizes; and a pg-script job that installs PostgreSQL 17 and pgvector
from the PostgreSQL apt repository and runs `scripts/pg.sh` end to end. The last two set a variable that turns their
tests' "skip if missing" into a failure, so a job can't pass by skipping.

**How is a shell script tested?**
`tests/test_pg_script.py` runs it on a throwaway cluster (`RECALL_PG_DATA` in a new directory, `RECALL_PG_PORT` a
free port): start, start again, status, url, the locale and code point order, localhost only, pgvector, stop, status
exit 3, a restart that keeps the data, and reset, plus reset refusing a directory that isn't a cluster. Its data
directory is under the repository because the server's socket lives there and pytest's temporary paths are longer
than a Unix socket path may be on macOS.

**Did CI find anything the Mac didn't?**
Yes, on the pg-script job's first run: `start` wrote initdb's log beside the data directory before creating it, so a
data directory whose parent didn't exist failed. Here `.scratch/` already existed (the review worktree was in it); a
fresh checkout doesn't have it. `start` now creates the parent, and the test starts in a parent that doesn't exist.

## Embeddings (day 2)

**Which model, and why?**
`BAAI/bge-small-en-v1.5`: 384 dimensions, 33.4M parameters, MIT-licensed, fast enough on a CPU (about 40 s for the
whole corpus). The deciding measurement was input length: `scripts/token_fit.py` counts every passage in the model's own
tokens. Passages are p50 187 and max 723 model tokens; bge-small reads 512, so 1 of 1,503 is truncated, while
all-MiniLM-L6-v2 (same tokenizer, 256 tokens) would truncate 369, a quarter of them.

**Why not an API model like OpenAI's embeddings?**
The project's rule is that everything runs on a laptop with no key, and tests and CI must never need one. A local
model also makes every vector reproducible from a pinned revision.

**What exactly is embedded?**
The heading path, page title first, then the chunk text (`passage_text`). A chunk often doesn't name its subject ("As
query parameters are not a fixed part of a path...") and its heading isn't chunk text, so the path adds what the
rendered page shows above it. Questions get the prefix bge's authors give for queries; passages get none.

**How do you keep vectors from two models apart?**
`corpus_settings.embedding_model` names the model and revision that made the stored vectors. Ingest with another model
sets every vector to NULL and embeds them all again (chunks keep their ids:
`test_a_new_model_embeds_every_chunk_again_and_keeps_the_chunks`), and `search` refuses to compare a question
embedded by one model with vectors from another (`ModelMismatch`).

**What is the hash embedder?**
A stub for tests with no dependencies: each token adds +1 or -1 to one of 384 buckets chosen by a hash, then the vector
is scaled to unit length. Texts that share words get similar vectors, so a test can predict a ranking exactly; it knows
nothing of meaning. It keeps PyTorch out of the main CI job; the real model has its own job.

**What if the model fails half way through an ingest?**
Embedding is inside ingest's transaction, so everything rolls back, vectors included
(`test_a_model_failing_part_way_leaves_the_previous_corpus_and_vectors`). So do a model of the wrong dimension, one
returning too few vectors, and a NaN (pgvector refuses it).

## Retrieval (day 2)

**How does full-text search work here?**
A generated `tsvector` column: the page title weighted A, the headings under it B, the text D, with the `english`
configuration (stemming, stop words), and a GIN index. The question is split at whitespace, each word becomes
`phraseto_tsquery`, and the words are ORed; matches are ranked by `ts_rank_cd`, normalized by log length.

**Why OR, and why a phrase per word?**
`plainto_tsquery` ANDs every word, and a question has more words than any passage shares, so AND finds little. The
phrase is for identifiers: the parser splits `response_model_exclude_unset` into four lexemes, and ORing them ranked
everything about response models above the section on that parameter (4th). As a phrase the four must be adjacent, and
it is 1st (`test_an_identifier_matches_as_written_not_as_its_parts`).

**Is that BM25?**
No. `ts_rank_cd` uses how often and how close together the words occur, with weights and length normalization, but no
inverse document frequency, so "return" counts as much as "HTTPException". The smoke test shows the cost: "How can I run
a function after returning the response?" doesn't find background-tasks.md in full-text's top 10. Day 4 measures
IDF weighting against it on labelled questions, instead of me guessing from a handful.

**Why does the generated column need a wrapper function?**
A generated column may only call IMMUTABLE functions, and `array_to_string` is STABLE (an element type's output can
depend on settings, like a timestamp's on DateStyle). For `text[]` nothing changes it, so `heading_text` declares that;
its SQL-standard body is bound at creation, so a later search path can't redirect it.

**How does vector search use the index?**
`ORDER BY embedding <=> question LIMIT n` is the shape the HNSW index serves; the row numbers are taken outside that
subquery so it stays that shape (`test_the_vector_query_can_use_the_hnsw_index` checks the plan).

**What is ef_search, and why do you set it?**
The size of HNSW's candidate list during a search; bigger is more accurate and slower. A scan returns at most ef_search
rows, and pgvector's default is 40, so a request for 50 came back with 40
(`test_vector_search_returns_k_rows_beyond_the_default_ef_search`). Each query raises it to its list length for its own
transaction.

**How does hybrid retrieval combine the two lists?**
Reciprocal rank fusion: each chunk scores the sum, over the lists it is in, of 1 / (60 + rank); the top 50 of each
list go in. Ranks rather than scores, because ts_rank_cd and cosine similarity aren't on one scale and RRF needs no
calibration. Both retrievers are CTEs of one query, so they see one snapshot and cost one round trip.

**What does RRF get wrong?**
It rewards agreement: a chunk both lists rank moderately beats one that only one list ranks high. In the smoke test,
for "Allow a frontend on another origin to call my API", frontend.md (full-text 6, vector 2: 0.03128) edges out
cors.md (4 and 4: 0.03125), though each retriever alone ranks cors.md first. And for "How do I return a custom status
code for an error?", vector has handling-errors.md 4th, but full-text doesn't have it, so hybrid drops it out of the
top 10.

**How do you know the hybrid SQL is right?**
A differential test: for random questions, k, candidate counts and RRF k, `hybrid` must equal a Python fusion of
`full_text` and `vector`'s own lists, ranks, scores and tie order included (`test_hybrid_is_the_fusion_of_the_two_lists`).
Mutating the SQL (the wrong rank in one term, the tie order) fails it.

**Why the 1 / (60 + rank) constant?**
k = 60 comes from the paper that introduced RRF (Cormack, Clarke and Buettcher, 2009); it keeps rank 1 from
dominating. It and the 50 candidates are defaults, not tuned; day 4's eval is where they get measured.

## Reranking (day 3)

**Why rerank at all, when you already fuse two retrievers?**
Both retrievers look at the question and each passage separately: one vector each, or word matches. A cross-encoder
reads the two together, so it can see that a passage answers the question rather than merely sharing its words. On the
smoke questions it moved query-params.md from hybrid's 4th to 1st and handling-errors.md from outside the top 10 to
5th (`tests/test_real_retrieval.py`).

**Why only the top 20?**
Each pair is one pass of the model. For 1,503 passages a question that's far too slow; for 20 it's a few hundred
milliseconds on this laptop. The cost is that it can only reorder what hybrid found: background-tasks.md isn't in
hybrid's top 20 for "How can I run a function after returning the response?", so reranking can't help. Day 4
measures the depth.

**Which model, and does the input fit?**
`cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, 91 MB), pinned to a revision like the embedder. It reads question
and passage as one 512-token sequence, so I counted pairs with its own tokenizer (`scripts/token_fit.py --reranker
--question ...`): 1 to 3 of 1,503 passages go over, 0.2% at most for a 24-word question; those lose their end.

**How is it tested without the model?**
`OverlapReranker` scores the share of the question's words in the text, so tests know the order exactly; `rerank`'s
own logic (best first, ties keep retrieval order, a wrong number of scores or a NaN refused) is tested with a reranker
that returns fixed scores. The real model has three obvious rankings and a check that the pinned `max_length` is
applied (a 16-token spec scores two passages that differ only past 16 tokens the same).

## Answers and citations (day 3)

**How does an answer get written?**
`answer.answer` numbers the top 5 passages, sends them and the question to the provider, and checks the JSON it gets
back. If the provider fails, the response isn't an answer, or no claim's citation checks out, it falls back to the
extractive answer and lists why in `problems`. If the model says the passages don't answer the question, there is no
answer at all.

**What makes a citation count?**
Its passage number must be one of those given (an `int`, checked with `type()` because `True` is an int in Python and
`2.0` isn't a passage) and its quote must be in that passage's text, exactly or with whitespace re-wrapped (models
change a newline into a space). At least 8 non-space characters, because one word matches almost anything. A citation
that counts becomes an exact span of the page: `document[start:end]` is the text the quote matched
(`test_every_quote_taken_from_a_passage_checks_out`). Claims without a citation that counts are dropped.

**Does that prove the answer is right?**
No. It proves every claim points at real text in a retrieved passage, so a reader can check it, and that the model
can't cite something it wasn't given. Whether the claim follows from the quote is faithfulness, which day 4 measures.

**Why not use Claude's own citations feature?**
It returns spans too, but it can't be combined with structured output, and the answer has to be checked the same way
whatever wrote it: a recording, another model, or a future provider. A substring check is simple, exact and
model-independent.

**There's no API key here. How is the live provider tested?**
Two ways, neither calling Claude. A fake client checks what `AnthropicProvider` asks for and how it treats a refusal
(`stop_reason: "refusal"`) and a truncated answer (`max_tokens`). And the real `anthropic` SDK is pointed at a local
HTTP server standing in for the API, so the request on the wire is checked: structured output with the answer's
schema, `fallbacks: "default"` with its beta header, the prompt (`test_the_real_sdk_sends_the_request_this_provider_means`).
What I haven't measured is how well Claude answers; that needs a key and is opt-in.

**What is a recording?**
A JSON file named after the SHA-256 of the request (system prompt, prompt and schema, canonical JSON), holding the
request and the response. The recorded provider replays it only if the stored request still hashes to the name, so an
edited recording is refused. Any change to the prompt or the passages is a different key, so a stale recording is
never replayed for the wrong question. Hand-written test recordings say `"model": "hand-written"`.

**How does the extractive answer pick sentences?**
It splits the passages into prose sentences (outside fenced code, no headings, list markers dropped; a sentence ends at
`.`, `!` or `?` and whitespace, except after `e.g.` and a few others), drops those under four words, scores them all
with the reranker against the question, and keeps the best three, each citing its own span.

**What went wrong with sentences and code?**
A code block longer than a chunk is split across chunks, so a chunk can start in the middle of one. Read alone, its
code lines are prose and its closing fence looks like an opening one, hiding the prose after it. I noticed a "sentence"
made of Python while spot-checking real chunks. `ask` now reads each passage's page, and sentences are found there;
129 of 1,503 real chunks changed (`test_a_chunk_that_starts_inside_a_code_block_is_read_in_its_document`).

## The API (day 3)

**How is the API built?**
FastAPI with plain (sync) handlers, which it runs on its thread pool. The models load once in the app's lifespan; a
psycopg pool of up to four autocommit connections serves requests, each holding one only for its database work: the
question is embedded before, and the reranker and the LLM run after the connection is back (day 4's fix of a day-3
review finding: four slow answers used to hold all four connections, and /search and /healthz got 503s meanwhile). `create_app` takes
a factory for its services, so tests pass the hash embedder, the overlap reranker and a pool into a test schema.

**Why run model calls one at a time?**
PyTorch already spreads one call over all the cores, so two calls at once would mostly take turns; and Hugging Face's
fast tokenizers have failed under concurrent use before ("Already borrowed"). I first wrote that they do fail, then
tested it: 320 calls from 8 threads without the lock raised nothing (tokenizers 0.23.3). So the lock is a precaution I
can defend on cost, not a fix for a failure I've seen; database work still runs in parallel.

**What status codes can it return?**
200; 422 for a bad request (an empty, blank or 2,001-character question, sizes out of bounds, passages above depth when
reranking, unknown fields); 503 when there's no database, no corpus, an old schema, or vectors from another model,
with a message that never includes the URL or the server's error text (either can hold a password).

**What bug did the API tests find?**
A question holding a lone surrogate (`"\ud800"` in JSON) is rejected by validation, but FastAPI's own 422 handler
echoes the input back, and that string can't be encoded as UTF-8, so the 422 became a 500. A custom handler returns
where and why without the input (`test_a_question_with_a_lone_surrogate_is_a_422_not_a_500`).

## Evaluation (day 4)

**How do you know retrieval works?**
`recall eval` runs 110 labelled questions through five systems (full-text, vector, hybrid, and the vector and hybrid
lists reranked) and reports recall@k (the share of questions with a gold chunk in the top k) and MRR over the top 20,
overall and per kind of question. CI's model job runs it on the real corpus with the real models and fails if any
system drops by more than one question's worth below the committed `eval/baseline.json`.

**Where did the questions come from?**
A Claude agent wrote them from the pages, read through `recall show`, with no access to the retrievers, their code or
their output, so none was picked or worded by what the system finds. Half name an identifier from the answer, half are
paraphrases that avoid the answer's distinctive words. They are synthetic and say so everywhere; they compare the
retrievers on this corpus, they don't predict what users would see. I read a sample and checked every gold quote
mechanically (it must occur exactly once in its page).

**Why are the labels quotes rather than chunk ids?**
Chunk ids change on every re-ingest and chunk boundaries change with the chunker. A quote of the answer's text is
resolved to whichever chunks overlap it when the evaluation runs (`gold_chunks`), so the labels survive re-chunking,
and a quote that is no longer in its page exactly once fails the run instead of silently scoring zero.

**What did it find?**
That my default is not the best pipeline. Vector search alone beats hybrid at every k (recall@10 0.727 vs 0.591; 15
questions only vector finds, none only hybrid finds, sign test p = 0.0001). Full-text search finds a gold chunk in its
top 10 for only 34% of questions, 62% even of the identifier ones, because ts_rank_cd has no IDF and the query ORs every
word, and RRF pulls its poor matches into the fused list. The reranker recovers some of it at rank 1 for hybrid, not
for vector.

**So why is hybrid still the default?**
Because the sweep of the fusion settings (`scripts/sweep_fusion.py`) says the fault is the full-text list, not the
fusion: no RRF k or candidate count beats vector alone on either half of the questions. Swapping the default to vector
on 110 synthetic questions, half of them written to defeat keyword search, would be tuning to the test. Day 5 measures
IDF weighting for full-text first, then picks the default from the ablation.

**Why a sign test?**
110 questions are few. Two systems' recall differs only by the questions one finds and the other misses; the exact sign
test asks how likely that split is if each such question were a coin flip. 15 to 0 is not chance; the reranker's 11 to
7 on vector at rank 1 could be.

**How did you avoid tuning on the test set?**
The sweep scores the odd-numbered and even-numbered questions separately: a setting picked on one half has to hold on
the other. Here the conclusion (lower k helps hybrid, nothing beats vector) held on both, and I changed no default.

**What does "faithfulness" mean here?**
For each claim, at least 75% of its content words must be in the passages it cites, and every exact term (numbers, code
spans, snake_case, CamelCase, dotted names) must appear there as written. It catches a wrong default or an invented
parameter. It can't catch a claim that negates its passage in the passage's own words; a test pins that limit.

**Your faithfulness score is 100%. Isn't that meaningless?**
On this machine, yes, and the README says so: with no recordings and no key every answer is extractive, and extractive
claims are the passages' own sentences. The run still checks the harness end to end, and the numbers that do mean
something are that 55.5% of answers were written from passages including a gold chunk and 50% cite one. With a key,
`RECALL_LLM=anthropic RECALL_RECORD=1 recall eval --answers` records model answers once and CI replays them.

**Can the gate pass a regression?**
A drop of one question per metric, by design: float differences between machines can move a near-tie. It refuses to
compare results under other settings (the question file's SHA-256, the corpus commit, the chunker and chunk count, the
models, depth, candidates, RRF k), so a change to any of those needs a new baseline committed with it, where a reviewer
sees the numbers move.

## The reviews

**How was the day-1 code reviewed?**
By three agents, one after the other, each on a fixed commit in its own git worktree, asked for a correctness review
against DESIGN.md: every bug reproduced as a failing test file (`tests/review/`, `tests/review2/`, `tests/review3/`), hypotheses listed
separately, and mutations (a mechanism removed or weakened) that no test noticed. I reproduced each finding here before
fixing it; each later review looked only at what changed since the one before, fixes included.

**What was the worst thing the first review found?**
Normalization found fences on the raw page and handled comments line by line afterwards, so the two disagreed: a
commented-out code example stayed in the page, and `<!--` written inside inline code (a page explaining comments)
silently deleted everything up to the next `-->`, headings included. The fix was structural: one pass in document
order with three states (code, comment or `<style>`/`<script>`, text), whichever starts first winning, as CommonMark
does, and code spans found over the whole paragraph.

**Why define `includes()` as "the files normalize reads"?**
The manifest is built from `includes()`, and `load` reads through `normalize()`. When they were two parsers, a page
with CRLF line ends or a comment on an include line had an include the manifest didn't list, so a corpus that
`verify` accepted couldn't be loaded. With one definition they can't disagree.

**Why did concurrent migrations fail under REPEATABLE READ, when there's a lock?**
The advisory lock was the transaction's first statement, and under REPEATABLE READ the snapshot is taken by the first
statement, before the lock wait: the waiter then ran on a snapshot from before the first migrator committed and
tried to create the same table. Waiting for a lock only helps if later statements see newer data, which is READ
COMMITTED, so `lock_read_committed` sets that level first. A database or role default is enough to trigger it.

**What did the second review find?**
Bugs in my fixes of the first review's findings, as second reviews have on every project so far. My fix for `# #`
emptied any title made only of `#` (`### ## #` should keep `##`). The one-pass normalizer could invent structure:
removing markup from `<b></b>```` left a fence opener that turned the rest of the page into code, and `<b>#</b> Title`
became a heading. Such lines now get a backslash escape, which also made normalization idempotent again. My database
error message printed the URL with its password. `<style-guide>` was taken for `<style>`, and many unclosed `<style`
openers took quadratic time. And `pg.sh` failed with a relative data directory, because the server resolves the socket
directory against its data directory. It also brute-forced the chunk rebalancing over 200,000 random cases and found
it correct.

**And a third review?**
Only on the second review's fixes, and it found seven more bugs in them: my front-matter guard looked at the page
rather than the output, my linear-time fix for `<style>` dropped too little, and my password redaction missed an
unescaped `@` and an unparseable URL's traceback. Each fix of a finding is new code, so I review it like new code.

**Did you disagree with the review anywhere?**
Once. It expected a raw include outside a fence to become a fenced block, because DESIGN.md said so. The documentation
build inserts such a file as Markdown, so the rendered page shows it as text; I changed DESIGN.md rather than the code,
and rewrote that test, saying why in its docstring.

**How do you know your tests test anything?**
The first review removed or weakened 37 mechanisms one at a time and 22 went unnoticed. I added a test for each and
then ran every one of those mutants again against the suite (copy aside, mutate, run, restore, compare). 21 of 23
were caught; the two that weren't (deleting a page exclusion) exposed a test that parametrized over
`source.EXCLUDED_PAGES` itself, so deleting an exclusion deleted its test case too. The test has its own list now. Two
of the review's survivors were dead code, which I removed, and one is `load`'s "only files in the manifest" check,
which `verify` makes unreachable.

## What went wrong on day 4

**What did the review of the evaluation find?**
Seven defects, two of them in the gate itself: a baseline with no scores passed any report, and a `--check` file
holding JSON `null` skipped the comparison and exited 0. A gate has to fail closed: anything it can't compare is now a
failure. The others: a quote copied with its newline slipped past the word-boundary rule I had added that morning;
overlapping repeats of a gold quote counted once (`str.count` skips overlaps); "1,500" was two terms to the
faithfulness check, so changing a default from 1,000 to 1,500 passed; and the sign test overflowed past 1,023
discordant questions.

**Which tests did it show were missing?**
19 of its 48 mutations left the suite green, for example: `rrf_k` and `candidates` never reaching the hybrid query (so
a baseline would record settings that weren't used), the faithfulness threshold moved to 0.6 or 0.9, chunks that only
touch a quote counted as gold. Each now fails a test; one survivor was redundant code, now removed.

**Did your fixes have bugs?**
Yes, again. A second review of only the fixes found two in my fix of the faithfulness terms: making "1,500" one term
cut "1,024MB" into a number and a unit, so a claim copying its passage word for word failed; and the passage side still
found "500" inside "1,500", because a word boundary treats "," and "." as boundaries. The fix looks for terms as whole
numbers too (`find_term`). Every multi-round review in this portfolio has found bugs in the previous round's fixes, so
I ask the second reviewer explicitly what each fix broke.

**What would you do differently?**
Write the gate's failure cases first: what a gate refuses matters more than what it accepts, and both of its bugs were
inputs I never imagined it receiving.

## What went wrong on day 1

**Tell me about a mistake you made.**
Two commits went in with a lint error because I chained the commit on `make check | tail`, and a pipeline's status
is its last command's, so `tail` succeeded while make failed. The tests had passed; ruff hadn't. The next commit fixed
each, and every later commit was chained on `make check` itself.

**And in the tests?**
I wrote expectations by hand that miscounted tokens: a fence's backticks are three tokens each side, so a fence I
thought had 10 tokens had 12. The tests failed, the arithmetic was wrong rather than the code, and the tests now assert
the count they depend on (`count_tokens(fence) == 12`) before relying on it.

## What went wrong on day 2

**What was the worst bug the review found?**
Vector lists came back short after any ingest that rewrote chunks: after a model change, a top 60 over 80 chunks
returned 31. An HNSW scan picks its ef_search nearest index entries and only then drops the dead ones, and the index
keeps entries for updated and deleted rows until VACUUM; a model change rewrites every vector, so half the nearest
entries were dead. My own model-change test hid it, because pgvector stores identical vectors as one element and my
stand-in "new model" gave the same vectors under a new name. The fix is pgvector 0.8's iterative scan
(`hnsw.iterative_scan = strict_order`), which keeps searching until the LIMIT is met, in distance order
(`tests/review_d2/test_review_d2_retrieval.py`).

**Tell me about a mistake you made.**
I raised the ef_search floor to 100 because my script said HNSW returned the exact top 10 there. It didn't: on 1,503
rows the planner switches from the index to a sequential scan at about ef_search 80 to 100, and my script checked the
plan only at 40, so "exact at 100" was an exact scan compared with itself. I found it with an EXPLAIN while looking at
something else. The script now forces each side's plan and checks it at every setting (real HNSW: 0.9987 at 100), the
floor is back at 40, and a test pins the top-10 plan to the index. That test passed with the wrong floor at first,
too, until it ran ANALYZE: a freshly loaded table has no statistics, and then the planner takes the index regardless.

**What else did the review find?**
That `search` checked the stored model in one snapshot and queried in another (an ingest in between could mix two
models' vectors; now one REPEATABLE READ transaction); that ef_search can't exceed 1,000, and a fresh session silently
fell back to 40 (lists are capped at 1,000); that a 20,000-word question made ts_rank_cd ask for 1.3 GB (each distinct
word once, the first 64); and four tracebacks the CLI now turns into one-line errors.

**Did your fixes have bugs?**
Yes; a second review of only the fixes found three. The 64-word cap didn't bound the query: one pasted list with no
spaces ("0,1,2,...,19999") is a single word whose phrase has 40,000 nodes, so the cap is now on nodes too (256, and a
word of more than 16 lexemes is skipped). `search` guarded its own transaction with `conn.autocommit`, which is still
true inside a caller's transaction; it checks the connection is idle now. And the iterative scan does give up, after
`hnsw.max_scan_tuples` (20,000) visited entries, dead ones included, so ingest now runs `VACUUM (ANALYZE)` after a
change, which also gives the planner statistics. It also showed the first review's race test passing for the wrong
reason after my fix (the race had moved before the check), so a new test races between the check and the query and
fails under READ COMMITTED.

**How did a test fail only sometimes?**
`test_the_indexes_exist` read `pg_indexes` for every schema in the test database, so while a review agent ran its own
tests there, extra rows appeared. I had guessed at a Hypothesis deadline; the second review found the real cause, and
the test now reads only its own schema.
