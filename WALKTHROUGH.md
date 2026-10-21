# Walkthrough

Questions an interviewer is likely to ask about Recall, with short answers grounded in the code. Day 1 covers the
corpus, normalization, chunking, storage and ingest; later days add their own sections.

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
file checks out. CI's corpus job downloads it fresh on every push and gets the same 1,505 chunks.

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
and heading path. It gives p50 142 and p95 253 tokens a chunk on this corpus. Day 2 checks it against the real
tokenizer, and day 5's ablation can vary it.

**Why no overlap?**
Cuts fall between blocks, not mid-sentence, and each chunk carries its page title and heading path, so the context
overlap usually buys is already there. Overlap would also make the same sentence appear in two passages and two
citations.

**What are the invariants?**
C1 a chunk is exactly `document[start:end]`, starting and ending on text; C2 chunks are in order without overlap; C3
every non-whitespace character outside heading lines is in a chunk; C4 every chunk has 1 to `max_tokens` tokens; C5 a
chunk lies in one section with that section's heading path; C6 a fence that fits is never split; C7 deterministic.

**How are they tested?**
Hypothesis builds random documents from known parts (headings, paragraphs, lists, fences containing `# comments`,
blank lines and runs of the fence character too short to close it), so the test knows where every heading and fence
is and checks each invariant against that, not against the chunker's own parse. The same checks run on every real
page at two sizes.

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
previous corpus (`test_a_failure_part_way_leaves_the_previous_corpus`). At 1,505 chunks it takes well under a second.
For a much bigger corpus I would load into a staging table and swap, to keep the transaction short.

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

## What went wrong on day 1

**Tell me about a mistake you made.**
Two commits went in with a lint error because I chained the commit on `make check | tail`, and a pipeline's status
is its last command's, so `tail` succeeded while make failed. The tests had passed; ruff hadn't. The next commit fixed
each, and every later commit was chained on `make check` itself.

**And in the tests?**
I wrote expectations by hand that miscounted tokens: a fence's backticks are three tokens each side, so a fence I
thought had 10 tokens had 12. The tests failed, the arithmetic was wrong rather than the code, and the tests now assert
the count they depend on (`count_tokens(fence) == 12`) before relying on it.
