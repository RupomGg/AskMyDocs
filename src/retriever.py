from embedder import embed_query
from vector_store import query_store

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Takes a natural language question, turns it into an embedding,
    and retrived the most relavent chunks from ChromeDB.

    """
    print(f"Embedding Question: '{question}' ...")

    #1. Embed the question which autometically uses RETRIEVAL_QUERY
    
    query_vector = embed_query(question)

    if not query_vector:
        print(f"failed to embed question '{question}'")
        return[]
    
    print(f"searching for top {top_k} matches...")

    #2. Search ChromeDB for the closes vector

    results = query_store(query_vector, top_k = top_k)

    return results

# --- simple testing Block ---
if __name__ == "__main__":
    
    test_question = "What is the XEB-FuseNet framework?"
    
    print(f"Testing retriever with question: {test_question}\n")
    results = retrieve(test_question)
    
    print(f"\nFound {len(results)} chunks.")
    for idx, r in enumerate(results):
        # Chroma's 'distance' metric: lower numbers mean it is a closer match!
        print(f"\n--- Result {idx + 1} (Distance: {r['similarity_distance']}) ---")
        print(f"Source: {r['metadata']['filename']} (Chunk {r['metadata']['chunk_index']})")
        print(f"Text snippet: {r['text'][:200]}...")
    
    