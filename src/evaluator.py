import hybrid_retriever
import citation_checker
import time
from hybrid_retriever import hybrid_retrieve
from generator import generate_answer
from citation_checker import check_citation


def evaluate_all(dataset_entries, keyword_seracher, reranker):
    results = []

    for idx, entry in enumerate(dataset_entries):
        print(f"\n---Evaluating [{idx+1}/{len(dataset_entries)}]: {entry['id']}---")
        question = entry['question']

        # 1. Retrieve and rerank
        hybrid_chunks = hybrid_retrieve(question, keyword_seracher, top_k=10)
        reranked_chunks = reranker.rerank(question, hybrid_chunks)
        top_3_chunks = reranked_chunks[:3]

        # 2. Generate answer
        print("Generating answer...")
        answer = generate_answer(question, top_3_chunks)

        # 3. Detect API/rate-limit failures BEFORE scoring
        # (generator.py returns "Sorry, I hit a rate limit..." on 429)
        is_api_failure = answer.startswith("Sorry,") 

        if is_api_failure:
            print("API failure detected for this entry — marking as ERROR, not scoring citations/coverage")
            passed_citation = None
            retrieval_coverage = None
        else:
            print("Checking citations...")
            passed_citation = check_citation(answer, top_3_chunks)

            print('Checking retrieval coverage...')
            retrieved_indices = [chunk['metadata']['chunk_index'] for chunk in top_3_chunks]
            expected_indices = entry.get('expected_chunk_indices', [])
            retrieval_coverage = True if not expected_indices else any(idx in expected_indices for idx in retrieved_indices)

        # 4. Store the Result
        result_record = {
            "id": entry["id"],
            "question": question,
            "expected_answer_summary": entry["expected_answer_summary"],
            "generated_answer": answer,
            "api_failure": is_api_failure,
            "passed_citation": passed_citation,
            "retrieval_coverage": retrieval_coverage,
            "human_grade": "Pending"
        }
        results.append(result_record)

        # Pause for 20 seconds to guarantee we stay under the 15 RPM limit
        time.sleep(20)  

    return results
