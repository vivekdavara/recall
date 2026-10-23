# Recordings

The recorded LLM provider (the default, `RECALL_LLM=recorded`) answers a question only if this directory holds a
recording of the exact prompt it would send: a file named after the SHA-256 of the request (`recall.llm.Request.key`),
holding the request and the response. Any other question gets the extractive answer.

`RECALL_LLM=anthropic RECALL_RECORD=1 recall ask "..."` records new ones with the live provider (it needs a key in the
environment). A recording whose `model` is `hand-written` was written by hand for a test, not by a model.
