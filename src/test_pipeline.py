from config import DOCUMENTS_DIR
from loader import load_documents
from chunker import chunk_text  
from embedder import embed_document_chunks
from vector_store import store_chunks
from retriever import retrieve
from generator import generate_answer
from keyword_search import KeywordSearcher
from hybrid_retriever import hybrid_retrieve
from reranker import Reranker


def run_pipeline():
    # 1. Loading raw text using loader
    print("\n--- 1. Loading Documents ---")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("No Documents found or failed to load")
        return
    
    print(f"Successfully loaded {len(documents)} documents.\n")

    # 2. Chunking documents
    print("--- 2. Chunking Documents ---")
    all_chunks = []
    for doc in documents:
        filename = doc["file_name"]
        raw_text = doc["raw_text"]
        
        chunks = chunk_text(raw_text=raw_text, filename=filename)
        all_chunks.extend(chunks)
        print(f"[{filename}] created {len(chunks)} chunks")

    print(f"Total chunks created across all documents: {len(all_chunks)}")

    # 3. Embed chunks (Converting text to numbers)
    print("\n--- 3. Generating Embeddings... (This calls Gemini, please wait) ---")
    embedded_chunks = embed_document_chunks(all_chunks)

    # 4. Building keyword_index
    print("\n--- 4. Building Keyword Index ---")
    keyword_searcher = KeywordSearcher()
    keyword_searcher.build_index(all_chunks)
    print("Keyword Index built successfully!")
    
    # 5. Store in ChromaDB
    print("\n--- 5. Storing in ChromaDB ---")
    store_chunks(embedded_chunks)

    #6 . initilazing reranker
    print("\n--- 6. Initializing Reranker ---")
    reranker = Reranker()

    # 7. Asking Questions!
    print("\n\n========================================================")
    print("          🌟 RAG PIPELINE FULLY OPERATIONAL 🌟            ")
    print("========================================================\n")
    
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        # This will pause the script and wait for you to type in the terminal
        question = input("\n👤 USER QUESTION: ")
        
        # If the user types exit, break the loop and end the script
        if question.lower().strip() in ['exit', 'quit']:
            print("Exiting RAG Pipeline. Goodbye!")
            break
            
        # Ignore empty questions (if you accidentally hit enter)
        if not question.strip():
            continue
            
        print("-" * 50)
        
        # 1. Retrieve top 10 closest chunks
        print("Running Hybrid Search...")
        hybrid_chunks = hybrid_retrieve(question, keyword_searcher, top_k=10)

        # 2. Rerank them based on absoulate relevance to the question
        print("Rrnaking Top Results...")
        reranked_chunks = reranker.rerank(question, hybrid_chunks)
        
        # 3 . taking the top 3 chunks to send to gemini 

        top_3_chunks = reranked_chunks[:3]

        # 4 . Pass the question and retrieved chunks to Gemini to generate the answer
        final_answer = generate_answer(question, top_3_chunks)

        print("\n🤖 AI ANSWER:")
        print(final_answer)
        print("========================================================")

if __name__ == "__main__":
    run_pipeline()

