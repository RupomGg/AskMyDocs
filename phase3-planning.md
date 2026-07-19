# Ask My Docs — Phase 3 Planning (Making it Shippable)

## Where we are
Phase 1 (ingestion → chunking → embedding → storage → retrieval → cited generation) and Phase 2 (hybrid retrieval, reranking, citation enforcement, versioned prompts) are both complete and battle-tested against real documents. Along the way you found and fixed two real citation-enforcement bugs, and identified a separate, still-open limitation: retrieval coverage for "best/worst across N categories" style comparison questions.

Phase 3 turns this from "a pipeline that works when I test it by hand" into "a pipeline whose quality is measured, not assumed."

## Learning framing for this phase
Everything so far has been verified by you, manually, reading each answer and judging it by eye. That doesn't scale, and more importantly, it means you have no way to know if a future code change (a new prompt version, a chunking tweak, a different top_k) made things better or worse — you'd have to re-read everything by hand again. Phase 3 replaces "I read the output and it looked right" with "a script scored the output against known-correct answers, and here is the number." This is the single biggest mindset shift in the whole roadmap: from vibe-checking to measuring.

## Non-goals for this phase
- No new retrieval or generation features (that was Phase 1/2) — Phase 3 measures what already exists
- No fine-tuning
- Full CI/CD (Part C below) is explicitly optional/stretch — treat Parts A and B as the real target

---

## Part A — Golden Dataset

### Concept to understand before building
A golden dataset is a fixed set of (question, correct answer, correct source) triples that you have personally verified against your actual documents. It exists so "is this answer right?" has an objective reference point instead of relying on your judgment fresh every time. You already have two entries sitting in your conversation history from Phase 2 debugging — that's not a coincidence; bugs you find while building often make the best eval cases, because you know exactly what correct and incorrect look like for them.

### Build order

**1. Design the schema first**
Decide the shape of one golden entry before writing any data. Suggested fields:
```json
{
  "id": "gd_001",
  "question": "What was the exact AUC score achieved for Atelectasis using Triple Fusion?",
  "expected_answer_summary": "Not explicitly stated; Dual Fusion AUC is 0.7746/0.775, Triple Fusion F1 is 0.8198, Accuracy is 0.7592",
  "expected_source_files": ["Final_Journal.pdf"],
  "expected_chunk_indices": [22, 24],
  "answer_type": "partial_decline",
  "notes": "Model should decline to state a Triple Fusion AUC rather than guessing"
}
```
`answer_type` is worth having as a category (e.g. `factual`, `partial_decline`, `full_decline`, `comparison`) — different question types need different scoring logic later, and tagging them now saves rework.

**2. `golden_dataset.json` (or split into a few files by topic)**
Populate with real entries, sourced two ways:
- **From bugs you already found** (highest value — you know ground truth cold):
  - The Cardiomegaly/Lung Opacity mixup case (expected: correctly distinguish the two, not conflate their AUCs)
  - The Atelectasis AUC case (expected: partial decline, correctly cites Dual Fusion figures, does not invent a Triple Fusion AUC)
- **New questions you write deliberately**, spanning your 5 documents:
  - Simple factual lookups (e.g. "What is Radwan's current job title?")
  - Numeric/table questions (e.g. specific AUC/F1 values per class)
  - Comparison/superlative questions (e.g. "which class performed worst" — you know this is a weak spot, so include several of these deliberately)
  - Out-of-scope questions with a known correct answer of "I don't have enough information..." (e.g. asking about a topic not in any of your 5 documents at all — tests whether the system correctly declines rather than hallucinating from outside knowledge)
  - Questions about the non-research documents too (Indico Asset Ledger features, CV content, MVP scope) — don't let the dataset skew entirely toward the one paper just because that's where the interesting bugs were

Target: start with 15-25 entries covering all 5 documents and all four `answer_type` categories, rather than rushing straight to 50-200. A small, carefully verified set teaches you more right now than a large, sloppy one — you can always add more later once the scoring script itself is trustworthy.

**Acceptance check:** for every entry, you personally re-verify the expected answer against the actual source document, not from memory of what you think it says (this is exactly the discipline that caught the Cardiomegaly/Lung Opacity bug in the first place).

---

## Part B — Automated Evaluation

### Concept to understand before building
"Faithfulness" here means: is the model's answer actually supported by the chunks it retrieved and cited? You've technically already built a faithfulness checker — `citation_checker.py` — for the generation step. Phase 3's evaluation script reuses that same idea, but runs it systematically across your whole golden dataset instead of one question at a time, and adds a second, different kind of check: **correctness against your known ground truth**, not just internal consistency.

This is an important distinction: `citation_checker.py` asks "does the answer's own citations check out?" The golden dataset eval asks "does the answer actually match what I, a human, verified to be true?" A model could pass the first check (every cited number is real) while still failing the second (it answered a different, wrong question, or missed the actual best-in-class answer) — which is exactly the retrieval-coverage gap you found earlier.

### Build order

**3. `evaluator.py`**
Purpose: run every golden dataset entry through the full pipeline (retrieve → generate) and score the result.

Function signature (approx): `evaluate_all(golden_dataset: list[dict]) -> list[dict]` — returns a per-entry result record.

For each golden entry:
- Run `hybrid_retrieve()` then `generate_answer()`, exactly as the real pipeline would
- Run the existing `check_citation()` — records whether the citation-enforcement layer flagged anything (this is your "internal consistency" signal)
- Compare the retrieved `chunk_indices` against `expected_chunk_indices` — did retrieval actually surface the right source material? (This directly measures the retrieval-coverage weakness you already found — now you'll have a number for it instead of a hunch.)
- Score answer correctness against `expected_answer_summary`. Start simple and human-inspectable rather than reaching for an automated similarity metric immediately:
  - Simplest first pass: print the generated answer next to the expected summary, and manually mark pass/fail/partial in the results file. This sounds low-tech, but it's the right first step — trust your own judgment before trusting an automated scorer, especially while you're still learning what "good" looks like for your own data.
  - Once you trust your manual judgments on a handful of cases, you *can* explore using an LLM call itself to judge correctness (a "judge model" pattern, which is genuinely how tools like Ragas work under the hood) — but treat that as a later refinement, not the starting point.

**4. `run_evaluation.py`** (the script you actually execute)
- Load `golden_dataset.json`
- Call `evaluate_all()`
- Print a summary: total entries, how many passed citation-check, how many had correct retrieval coverage, how many were judged correct
- Save detailed per-entry results to a file (e.g. `eval_results.json`) so you can review failures afterward without re-running the whole pipeline

**Acceptance check:** run this against your current pipeline and get real numbers — e.g. "18/20 passed citation check, 15/20 had correct retrieval coverage, 14/20 judged correct." These numbers becoming your baseline is the actual milestone here — not any particular score, just *having* a measured baseline you didn't have before.

---

## Part C — CI/CD Integration (optional stretch goal)

### Concept to understand before building
The idea: wire `run_evaluation.py` into a GitHub Actions workflow so that every time you push code, the evaluation runs automatically, and if your pass rate drops below a threshold, the pull request/commit gets flagged. This is genuinely how production ML/RAG teams prevent silent quality regressions — but it's also the most infrastructure-heavy, least conceptually new part of Phase 3.

### Why this is marked optional for you specifically
You're a solo developer learning end-to-end RAG concepts, not shipping to a team that needs an automated gate. The learning value of understanding *why* CI-gated evaluation matters is high; the practical value of actually setting up GitHub Actions for a portfolio project you control alone is lower. Treat this the same way we treated structured-output citations in Phase 2 — worth doing if you have the time and curiosity, not worth blocking on.

### If you want to try it
**5. `.github/workflows/eval.yml`**
- Trigger on every push
- Set up Python, install dependencies
- Run `run_evaluation.py`
- Add a simple check: if pass rate falls below a hardcoded threshold (e.g. 70%), exit with a non-zero status code so GitHub marks the run as failed
- Note: this will call the Gemini API on every push, which costs real API calls against your free-tier quota — worth being deliberate about how often this actually runs (e.g. not on every single commit if you're committing frequently while debugging)

---

## Definition of done for Phase 3
- A golden dataset of 15-25+ manually-verified entries exists, covering all 5 documents and multiple question types (factual, numeric, comparison, decline-expected)
- `run_evaluation.py` produces a real, reproducible score against that dataset
- You know, with numbers rather than a general impression, roughly how often your pipeline is faithful, how often retrieval actually surfaces the right chunks, and how often the final answer is judged correct
- You can explain what each of the three scoring dimensions (citation-consistency, retrieval-coverage, answer-correctness) measures and why they're different from each other
- (Stretch) CI workflow exists and runs the evaluation automatically on push

## Instruction for the coding agent
Build Part A fully (a real, hand-verified golden dataset) before writing any evaluation code — a scoring script is useless without trustworthy ground truth to score against. Keep the correctness-scoring approach in Part B manual/human-in-the-loop at first; do not reach for an automated LLM-judge pattern until the person has built confidence in what correct scoring looks like by doing it manually first. Do not build Part C unless explicitly asked to continue past Parts A and B. Ask before adding any new dependency (e.g. Ragas, if the person later wants to try an established evaluation framework instead of hand-rolled scoring).
