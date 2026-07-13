# Ask My Docs — Phase 2 Planning (Production Quality)

## Where we are
Phase 1 is complete and verified: ingestion, chunking, embeddings (Gemini), storage (ChromaDB), retrieval, and citation-labeled generation (Gemini) all work end-to-end against real documents. One documented limitation was found: dense tables can be split across chunk boundaries, causing the LLM to occasionally mix up numbers between adjacent tables — a real cross-chunk attribution error, not a formatting bug.

## Learning framing for this phase
Each Phase 2 topic below exists to fix a specific, real weakness — not to add complexity for its own sake. Before writing code for each piece, understand *why* it exists and *what failure it prevents*, not just how to wire it up. Where possible, deliberately break something first (feed a bad question, a keyword-heavy question, a question you know isn't answerable) so you can see the failure with your own eyes before adding the fix — this is the same "break it before you trust it" habit from Phase 1.

Suggested order: **Citation Enforcement first**, since it directly targets the table-mixup bug you already found and documented — you have a real, concrete test case in hand rather than a hypothetical one. Hybrid retrieval and reranking come next since they improve what gets retrieved in the first place. Prompt architecture is a structural cleanup that can happen alongside any of the above.

## Non-goals for this phase (still not yet — that's Phase 3)
- No golden dataset / automated evaluation scoring
- No CI/CD integration
- No fine-tuning or model training of any kind

---

## Part A — Citation Enforcement (do this first)

### Concept to understand before coding
Right now, `generator.py` *asks* the model (via prompt instructions) to only use provided chunks and cite sources — but nothing actually *checks* whether it did. Citation enforcement adds a verification step after generation: does each claim in the answer actually trace back to text that exists in the retrieved chunks? If not, the system should flag it or decline, rather than silently presenting an unverified claim as fact.

This is the direct fix for what happened with the Cardiomegaly/Lung Opacity mixup — the model claimed a chunk supported a number it didn't actually support.

### Build order

**1. Reproduce the bug as a fixed test case**
Before writing enforcement logic, write down (in a small script or plain text file) the exact known-bad case: question, the chunks that were retrieved, and the incorrect answer that came back. This becomes your first real test — you'll know your enforcement logic is working when it catches *this specific case*, not just when it "seems to work."

**2. `citation_checker.py`**
Purpose: given the model's generated answer and the chunks it was supposed to use, check whether cited claims are actually supported.
Approach for a first, learnable version (don't reach for anything fancy yet):
- Parse out the citations the model included (e.g. regex for the `(Source: filename, Chunk N)` pattern you already prompt for)
- For each citation, pull the actual chunk text it claims to reference
- Do a simple overlap/keyword check: does the cited chunk actually contain the number/fact the answer claims near that citation? (This can be as simple as checking whether key numbers mentioned in the answer literally appear in the cited chunk's text — a blunt but genuinely instructive first pass.)
- Flag mismatches rather than silently passing them through

Acceptance check: run this against your fixed test case from step 1 — it should flag the Cardiomegaly/Lung Opacity mismatch as unsupported.

**3. Wire into `generator.py`**
After generation, run the answer through `citation_checker.py`. If a citation doesn't check out, either:
- Append a visible caveat to the answer ("Note: one or more citations could not be verified against source text"), or
- Regenerate once with a stricter instruction, or
- Return a decline message for that specific claim
Pick the simplest of these to start (the caveat) — you can make it stricter once you see how often it actually fires on real questions.

**Learning checkpoint:** Can you explain, without looking at the code, why keyword/overlap checking is a blunt instrument, and what kinds of correct answers it might incorrectly flag as "unsupported" (e.g. an answer that correctly paraphrases a chunk without repeating its exact numbers)? This tension — precision vs. recall in your enforcement check — is worth sitting with before automating it further.

---

## Part B — Hybrid Retrieval

### Concept to understand before coding
Vector/semantic search is good at matching *meaning* but can miss exact terms — a specific model name, an exact number, an acronym like "XEB-FuseNet" — because embeddings blur precise wording into a general "vibe." **BM25** is a classic keyword-matching algorithm (from traditional search engines, pre-dating embeddings entirely) that's strong exactly where vector search is weak: exact term matches. Hybrid retrieval runs both and merges the results, so you get the best of both.

### Build order

**4. Deliberately find a failure case first**
Ask your current system (vector-only) a question that hinges on an exact term or number appearing in your docs (e.g. a specific model name mentioned only once, or an exact date). See if pure vector search retrieves the right chunk. If it already does, try a harder one — the point is to *feel* where vector search alone struggles before adding BM25 as the fix.

**5. `keyword_search.py`**
Purpose: implement BM25 search over the same chunk text you already have stored.
- A simple library like `rank_bm25` works fine here — you don't need to hand-write BM25 itself to learn the concept, but do read how it scores (term frequency + inverse document frequency + length normalization) so it's not a black box.
- Index your chunk texts (same chunks already in ChromaDB) separately for keyword search.

**6. `hybrid_retriever.py`**
Purpose: combine vector search results (`retriever.py`) with keyword search results (`keyword_search.py`) into one ranked list.
- Simplest approach: run both searches, take the union of results, and combine scores (a common simple method is Reciprocal Rank Fusion — worth reading about, not hard to implement by hand).
- Acceptance check: rerun your failure case from step 4 — does the exact-term question now retrieve the right chunk?

---

## Part C — Reranking

### Concept to understand before coding
Your initial retrieval (vector + keyword) is optimized for speed — finding a good-enough shortlist fast across potentially thousands of chunks. A **cross-encoder reranker** is slower per-item but much more precise: it looks at the question and a *specific* chunk together (rather than comparing pre-computed embeddings), so it can catch nuanced relevance that similarity search misses. You only run it on your small shortlist (e.g. top 10-20), not the whole database — that's why it's affordable despite being slower.

### Build order

**7. `reranker.py`**
- Use a small, free cross-encoder model (e.g. via `sentence-transformers`'s `CrossEncoder` class) — no API key needed, runs locally.
- Function: takes the question + the hybrid retrieval shortlist, re-scores each chunk, returns the top-k re-sorted by the new scores.
- Acceptance check: compare the pre-rerank vs. post-rerank order on a few real questions — does the reranked top result look more directly relevant by eye?

---

## Part D — Prompt Architecture

### Concept to understand before coding
Right now your prompt template lives as an f-string inside `generator.py`. As you iterate (which you will, especially once citation enforcement is in place), you'll want to change instructions without touching pipeline code, and you'll want to know what prompt version produced what answer. Treating prompts like versioned config is a real production practice, not just tidiness.

### Build order

**8. `prompts/answer_prompt_v1.txt` (or a small `prompts.py` with versioned templates)**
- Move the prompt text out of `generator.py` into its own file.
- Include a version identifier (even just a filename suffix like `_v1`) so future changes are traceable.
- `generator.py` now reads/loads the template rather than hardcoding it inline.

---

## Definition of done for Phase 2
- Citation enforcement catches the known Cardiomegaly/Lung Opacity test case, and you understand its precision/recall tradeoffs
- Hybrid retrieval demonstrably retrieves at least one exact-term case that vector-only search missed
- Reranking visibly reorders a shortlist toward more relevant chunks on a real example
- Prompt template lives outside the Python code, versioned
- You can explain each piece's purpose and tradeoffs out loud, without the IDE's AI explaining it back to you

## Stretch Goal — Structured Output for Citations (optional, not required for Phase 2 completion)

### The lesson this comes from
While building citation enforcement, the regex-based parser broke twice from the model simply varying how it phrased multi-chunk citations ("Chunk 23, 24" vs "Chunk 23, Chunk 24" vs potentially "Chunk 23 and 24"). Each time, the fix was to make the regex more tolerant — but this is fundamentally a losing game: an LLM's prose phrasing can vary in more ways than any fixed regex can anticipate, so parsing free-text output after the fact will always be somewhat fragile.

### The more robust alternative
Instead of asking the model to embed citations inline in prose and then regex-parsing them back out, ask it to return a **structured response** — e.g. JSON with an explicit schema — where citations are a proper list of objects rather than text to be pattern-matched.

Example shape:
```json
{
  "answer": "The Triple Fusion framework achieved its best AUC for Edema...",
  "claims": [
    {
      "text": "AUC: 0.9052",
      "source_file": "Final_Journal.pdf",
      "chunk_indices": [23, 24]
    }
  ]
}
```

This removes the parsing-fragility problem entirely — `chunk_indices` is already a real list of integers, no regex needed, no ambiguity about separators or phrasing. The tradeoff: you lose some of the model's natural prose flow unless you also render the structured claims back into readable text yourself, and you're now relying on the model reliably following a JSON schema (which Gemini and most modern LLMs support well via structured output / JSON mode, but it's a different failure mode to handle — malformed JSON — rather than eliminating failure modes entirely).

### If you want to try this later
- Gemini's API supports a `response_schema` / structured output mode — worth reading the docs on this specifically rather than assuming it works like plain JSON-mode prompting
- Keep `citation_checker.py`'s core verification logic (checking whether claimed numbers appear in the cited chunk text) — only the *parsing* step changes, not the verification concept
- This is a good "if I have extra time" upgrade, not a requirement — the current regex-based approach, now fixed twice, works for the test cases you have. Don't rebuild working code purely for elegance; revisit this only if regex breakage becomes a recurring time sink.

## Instruction for the coding agent
Build one part (A, B, C, or D) at a time, fully, before starting the next — do not scaffold all four simultaneously. Keep implementations simple and inspectable (e.g. blunt keyword-overlap citation checks, off-the-shelf small libraries for BM25/reranking) rather than reaching for heavyweight abstractions. Do not introduce LangChain or a full RAG framework. Ask before adding any dependency not already agreed (rank_bm25, sentence-transformers CrossEncoder).

