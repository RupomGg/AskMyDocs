# Ask My Docs — Phase 1 Planning

## Goal

Build the foundational RAG pipeline: ingest documents → chunk them → (embeddings/storage come after this phase, not yet). By the end of Phase 1, we should be able to load a folder of documents and produce clean, well-formed, correctly-sized text chunks with metadata, verified by manual inspection.

## Non-goals for Phase 1 (do not build yet)

- No embeddings
- No vector database
- No retrieval logic
- No LLM calls
- No hybrid search, reranking, or evaluation — that's Phase 2/3

## Tech stack for this phase

- Python 3
- `pypdf` for PDF text extraction
- `tiktoken` for token counting (not character counting)
- Plain functions, no frameworks (no LangChain) — the point is to understand the mechanics directly

## Test corpus

4–6 personal documents (CV, project write-ups, research papers) placed in a local `documents/` folder. Mixed lengths — at least one longer document (10+ pages) to properly exercise chunk-boundary behavior.

## Build order

### 1. `config.py`

Purpose: central place for constants, so nothing is hardcoded elsewhere.
Contents:

- `DOCUMENTS_DIR` — path to the folder of source documents
- `CHUNK_SIZE` — target tokens per chunk (start at 600)
- `CHUNK_OVERLAP` — overlap tokens between consecutive chunks (start at 100)

Acceptance check: file exists, values are easy to tweak in one place.

### 2. `loader.py`

Purpose: read raw text out of every supported file in `DOCUMENTS_DIR`.
Function signature (approx): `load_documents(folder_path) -> list of {filename, raw_text}`
Behavior:

- Iterate over files in the folder
- For PDFs, extract text with `pypdf`
- For `.md`/`.txt`, just read directly
- Return one record per file: filename + full extracted text

Acceptance check: for each test document, print the first ~500 characters and manually confirm the text isn't garbled, missing sections, or full of extraction artifacts (common PDF issue — check tables/columns especially).

### 3. `chunker.py`

Purpose: split each document's raw text into overlapping chunks sized by token count.
Function signature (approx): `chunk_text(raw_text, chunk_size, overlap) -> list of chunks`
Behavior:

- Use `tiktoken` to count tokens (do not approximate by character count or word count)
- Produce chunks of `CHUNK_SIZE` tokens with `CHUNK_OVERLAP` tokens shared between consecutive chunks
- Each chunk should carry metadata: source filename, chunk index (position within the document)

Acceptance check: total token counts per chunk are close to the target size; overlapping regions are visibly present when printed; no chunk is empty or absurdly short/long.

### 4. `test_pipeline.py` (throwaway script, not core code)

Purpose: run `loader.py` → `chunker.py` end-to-end on the real test corpus and print results for manual review.
What to look for:

- Does chunking break mid-sentence in an unreasonable way?
- Is the overlap actually preserving context across chunk boundaries (read two consecutive chunks — does the second sensibly continue from the first)?
- Are chunk sizes roughly consistent with `CHUNK_SIZE`?
- Does metadata correctly track which file and position each chunk came from?

## Definition of done for Phase 1

- Can point the pipeline at any folder of PDFs/text files and get back a list of chunks with metadata
- Chunks have been manually reviewed against the real test corpus and look coherent
- No embedding, storage, or retrieval code has been written yet — that's the next phase
- Every function in `loader.py` and `chunker.py` can be explained line-by-line without relying on the IDE's AI to re-explain it

## Explicit instruction for the coding agent

Write minimal, readable code. Favor plain Python and clear variable names over cleverness or premature abstraction. Do not introduce LangChain, LlamaIndex, or any RAG framework in this phase — the goal is to understand raw mechanics first. Do not implement embeddings, vector storage, or retrieval — stop at chunking. Ask before adding any dependency not listed in the tech stack above.
