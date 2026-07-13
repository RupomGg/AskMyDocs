from retriever import retrieve
from keyword_search import KeywordSearcher

def hybrid_retrieve(query:str, keyword_searcher: KeywordSearcher, top_k: int = 5) -> list[dict]:
    """
    Runs both vector search and keyword serach, then complies the results.
    using Reciprocal Rank fusion (RRF) algorithm

    """

    #1. Retrieve results from both search functions
    vector_results = retrieve(query, top_k = top_k)
    keyword_results = keyword_searcher.search(query, top_k = top_k)

    #2. Combineing using Reciprocal Rank fusion algorithm
    chunk_scores = {}
    chunk_data = {}


    def add_to_rrf(results,k_constant = 60):
        for rank, chunk in enumerate(results):
            # --- BUG FIX: Normalize chunks from BM25 to match ChromaDB's nested format ---
            if 'metadata' not in chunk:
                chunk = {
                    'text': chunk['text'],
                    'metadata': {'filename': chunk['filename'], 'chunk_index': chunk['chunk_index']}
                }
                
            # Unique Id for each chunk to match them up
            chunk_id = f"{chunk['metadata']['filename']}_chunk{chunk['metadata']['chunk_index']}"

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0.0
                chunk_data[chunk_id] = chunk

            # RRF formula : 1/(60+rank) -- Higher is better but its inverse so 1/ (60+rank)

            chunk_scores[chunk_id] += 1.0 / (k_constant + rank)
    
    # assigning points for both lists

    add_to_rrf(vector_results)
    add_to_rrf(keyword_results)

    # 3. ranking combined RRF Score
    sorted_chunks = sorted(chunk_scores.items(),key=lambda item: item[1], reverse = True)

    #4. Returning top_k chunk dictonaries
    
    final_results= []

    for chunk_id, score in sorted_chunks[:top_k]:
        final_results.append(chunk_data[chunk_id])

    return final_results