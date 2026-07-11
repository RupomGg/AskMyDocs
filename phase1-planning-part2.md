# Ask My Docs — Phase 1 Planning (Part 2: Storage & Retrieval)

## Where we are

Done: document ingestion (`loader.py`) and chunking (`chunker.py`), verified working end-to-end via `test_pipeline.py`, with the filename/text key-mismatch bug fixed and an overlap>=chunk_size validation guard added.

Remaining Phase 1 steps: turn chunks into embeddings, store them in a vector database, then build retrieval + citation-backed answering.

## Non-goals for this part (still not yet — that's Phase 2/3)

- No hybrid (BM25 + vector) search
- No reranking
- No citation-enforcement / hallucination guardrails
- No evaluation harness
- No LangChain or other RAG framework

## Embedding provider: Google Gemini (decided)

Using Google AI Studio's free-tier Gemini API — not the Google AI Pro subscription (that's a separate consumer chat product and does not include API access).

- Model: `gemini-embedding-001` (stable, generally available — avoid the `-preview` variants for this project so behavior doesn't shift under you mid-build)
- SDK: `google-genai` (the current official Python package — install as `google-genai`, import as `from google import genai`)
- Auth: API key from Google AI Studio (aistudio.google.com/apikey), set as an environment variable, e.g. `GEMINI_API_KEY`
- Endpoint used under the hood: `embed_content`
- Output: 3072-dimensional vectors by default. Gemini supports Matryoshka Representation Learning (MRL), meaning you can request a smaller `output_dimensionality` (e.g. 768) to save storage/compute with minor quality tradeoff — not required for Phase 1, but good to know it exists.
- Task type parameter: Gemini's `embed_content` accepts a `task_type` argument that tunes the embedding for its intended use. For this project:
  - Use `RETRIEVAL_DOCUMENT` when embedding chunks going into storage
  - Use `RETRIEVAL_QUERY` when embedding an incoming user question
    This distinction matters — embedding a question and a document chunk slightly differently for their respective roles generally improves retrieval quality. Do not use the same task type for both.
- Rate limits: free tier has per-minute and per-day request caps that vary by model — check your AI Studio dashboard directly rather than assuming a fixed number, since these change.

## Build order

### 5. `embedder.py`

Purpose: convert chunk text (or a query) into a Gemini embedding vector.
Function signatures (approx):

- `embed_document_chunks(chunks: list[dict]) -> list[dict]` — takes chunks from `chunker.py`, adds an `"embedding"` key to each using `task_type="RETRIEVAL_DOCUMENT"`, returns the enriched list
- `embed_query(question: str) -> list[float]` — embeds a single question using `task_type="RETRIEVAL_QUERY"`, returns the vector directly

Behavior:

- Initialize the `genai.Client()` once, not per call
- Loop over chunks and call `embed_content` for each (batching can come later if needed — a simple loop is fine for a handful of test documents)
- Handle errors explicitly: rate limit (429), auth errors (invalid/missing key), and any malformed response — this ties back to your original "handle errors properly" goal from Phase 1 step 1

Acceptance check: print the embedding vector's length for a couple of chunks — should consistently be 3072 (or whatever `output_dimensionality` you set, if you choose to override the default). If the length varies between calls, something is wrong.

### 6. `vector_store.py`

Purpose: store embedded chunks in ChromaDB and support similarity search over them.
Two functions (approx):

- `store_chunks(chunks: list[dict])` — adds chunks (text + embedding + metadata: filename, chunk_index) into a persistent Chroma collection
- `query_store(query_embedding, top_k=5) -> list[dict]` — returns the top-k most similar stored chunks, with their metadata, for a given query embedding

Acceptance check: store your real chunks, then manually query with a known phrase's embedding (something you know appears in one specific document) and confirm the top result is actually the right chunk from the right file.

### 7. `retriever.py`

Purpose: tie together "user asks a question" → "question gets embedded via `embed_query`" → "top-k chunks come back from the vector store."
Function signature (approx): `retrieve(question: str, top_k=5) -> list[dict]`
Behavior:

- Call `embed_query` from `embedder.py` (critical: this must use `RETRIEVAL_QUERY`, while stored chunks used `RETRIEVAL_DOCUMENT` — mismatching these hurts retrieval quality even though both still produce 3072-dim vectors)
- Call `query_store` from `vector_store.py` and return results

Acceptance check: ask a question you know the answer to from your test corpus (e.g. "what programming languages does this person know") and confirm the retrieved chunks actually contain that information.

### 8. `generator.py`

Purpose: take the retrieved chunks + the original question, build a prompt, call an LLM, and return an answer that cites its sources.

Provider decision: no Claude API key available, so use Gemini for generation as well — same `google-genai` SDK and same API key already set up for embeddings. Model: `gemini-2.5-flash` (free-tier eligible as of this writing). This means one provider, one API key, for both embeddings and generation in Phase 1.

Behavior:

- Construct a prompt that includes: the question, the retrieved chunks (labeled with their source filename), and an explicit instruction to answer only from the provided chunks and cite which file/chunk supported each claim
- Call `client.models.generate_content(model="gemini-2.5-flash", ...)` using the same `genai.Client()` instance pattern as `embedder.py`
- Handle errors explicitly and distinctly:
  - **429 / quota exceeded** — free-tier rate limits are real and can be hit during testing (limits are per Google Cloud project, not per API key, and Google has cut free quotas before with little warning). Catch this specifically and print a clear message (e.g. "Rate limit hit — wait and retry" or implement a short backoff-and-retry) rather than letting the program crash with a raw exception.
  - **Model unavailable / billing required** — if `gemini-2.5-flash` ever isn't reachable on the free tier (model access can shift), fail with a clear, specific error message naming the model, so it's obvious what to check (rather than a generic "request failed").
  - **Timeouts** — handle same as any network call.
- Return the answer text along with which chunks were used

Before building this file: do one manual sanity check first — in AI Studio's Playground UI (not code), send a test prompt to `gemini-2.5-flash` and confirm it responds without a billing/quota error. This confirms free-tier access before writing generation code around it.

Note: since both embedding and generation now go through Gemini, `embedder.py` and `generator.py` will look similar (same client setup, same SDK) but call different methods — `embed_content` vs `generate_content` — and serve different purposes. Keep them as separate files/functions regardless, since they're conceptually distinct steps in the pipeline.

Acceptance check: ask a few questions where you already know the right answer from your test documents, and confirm both (a) the answer is factually correct and (b) the citation actually points to the real source.

### 9. Update `test_pipeline.py`

Extend the existing script to run the full chain: load → chunk → embed → store → retrieve → generate, for a few hardcoded test questions. Print the question, the retrieved chunk sources, and the final cited answer for manual review.

## Definition of done for Phase 1 (complete)

- Can point the pipeline at a folder of documents and ask natural-language questions about them
- Every answer includes a citation back to the specific source file/chunk
- Manually verified against your real test corpus: answers are correct, citations are accurate
- You can explain, line by line, how a question turns into an embedded query, how retrieval finds relevant chunks, and how the final answer gets constructed — without relying on the IDE's AI to explain it back to you

## Instruction for the coding agent

Keep it minimal and readable — plain Python, no RAG frameworks yet. Use the `google-genai` SDK for both embeddings (`gemini-embedding-001`) and answer generation (`gemini-2.5-flash`) — one provider, one API key, for this phase. Do not add hybrid search, reranking, or evaluation logic in this phase. Do not silently swallow errors from the embedding or LLM API calls — surface them clearly, with distinct handling for rate-limit/quota errors versus other failures. Ask before adding any dependency not already agreed (tiktoken, pypdf/markitdown, chromadb, google-genai).
