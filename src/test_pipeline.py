from config import DOCUMENTS_DIR
from loader import load_documents
from chunker import chunk_text  
from embedder import embed_document_chunks
from vector_store import store_chunks
from retriever import retrieve
from generator import generate_answer

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

    # 4. Store in ChromaDB
    print("\n--- 4. Storing in ChromaDB ---")
    store_chunks(embedded_chunks)

    # 5. Asking Questions!
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
        
        # Retrieve top 3 closest chunks
        retrieved_chunks = retrieve(question, top_k=3)
        
        # Pass the question and retrieved chunks to Gemini to generate the answer
        final_answer = generate_answer(question, retrieved_chunks)
        
        print("\n🤖 AI ANSWER:")
        print(final_answer)
        print("========================================================")

if __name__ == "__main__":
    run_pipeline()

