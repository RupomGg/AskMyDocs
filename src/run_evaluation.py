import json
import os
from config import DOCUMENTS_DIR
from loader import load_documents
from chunker import chunk_text
from embedder import embed_document_chunks
from vector_store import store_chunks
from keyword_search import KeywordSearcher
from reranker import Reranker
from evaluator import evaluate_all

def main():
    print("Initializing pipeline for evaluation...")
    
    # 1. Loading raw text using loader
    print("\n--- 1. Loading Documents ---")
    documents = load_documents(DOCUMENTS_DIR)
    
    # 2. Chunking documents
    print("--- 2. Chunking Documents ---")
    all_chunks = []
    for doc in documents:
        filename = doc["file_name"]
        raw_text = doc["raw_text"]
        chunks = chunk_text(raw_text=raw_text, filename=filename)
        all_chunks.extend(chunks)

    # 3. Embed chunks
    """print("\n--- 3. Generating Embeddings... ---")
    embedded_chunks = embed_document_chunks(all_chunks)"""

    # 4. Building keyword_index
    print("\n--- 4. Building Keyword Index ---")
    keyword_searcher = KeywordSearcher()
    keyword_searcher.build_index(all_chunks)
    
    """# 5. Store in ChromaDB
    print("\n--- 5. Storing in ChromaDB ---")
    store_chunks(embedded_chunks)
    """

    # 6. Initialize Reranker
    print("\n--- 6. Initializing Reranker ---")
    reranker = Reranker()
    
    # --- EVALUATION LOGIC START ---
    
    # 7. Load the Golden Dataset
    dataset_path = "data/golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        entries = data["entries"] 
        
    # --- NEW RESUME LOGIC ---
    results = []
    completed_ids = set()
    results_path = "data/eval_results.json"
    
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            for r in existing_results:
                # Check if it was a rate limit failure. 
                # (We check the string directly in case 'api_failure' wasn't saved in the old run)
                is_api_failure = r.get("api_failure", r.get("generated_answer", "").startswith("Sorry,"))
                
                if not is_api_failure:
                    results.append(r)
                    completed_ids.add(r["id"])
                    
    print(f"\nFound {len(completed_ids)} already completed entries. Skipping them!")
    
    # Filter out the entries we already completed
    remaining_entries = [e for e in entries if e["id"] not in completed_ids]
    
    # 8. Evaluate the remaining dataset 
    if remaining_entries:
        print(f"Starting evaluation of the {len(remaining_entries)} remaining entries...")
        new_results = evaluate_all(remaining_entries, keyword_searcher, reranker)
        results.extend(new_results)
    else:
        print("All entries have already been successfully evaluated!")
    
    # 9. Calculate automated metrics
    scoreable = [r for r in results if not r.get("api_failure")]
    failures = [r for r in results if r.get("api_failure")]

    total = len(scoreable)
    citation_passed = sum(1 for r in scoreable if r.get("passed_citation"))
    coverage_passed = sum(1 for r in scoreable if r.get("retrieval_coverage"))

    print("\n========================================================")
    print("                 📊 EVALUATION SUMMARY 📊                 ")
    print("========================================================")
    print(f"API Failures (Excluded from scoring): {len(failures)}")
    
    if total > 0:
        print(f"Faithfulness (Passed Citation): {citation_passed}/{total} ({(citation_passed/total)*100:.1f}%)")
        print(f"Retrieval Coverage (Found Expected Chunk): {coverage_passed}/{total} ({(coverage_passed/total)*100:.1f}%)")
    else:
        print("No successful generations to score.")
        
    print("\nNote: 'Answer Correctness' requires manual grading.")
    print("========================================================\n")
    
    # 10. Save the detailed results
    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("Detailed results saved to data/eval_results.json")

if __name__ == "__main__":
    main()
             