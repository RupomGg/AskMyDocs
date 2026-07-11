import chromadb
from pathlib import Path

# creating local db inside a folder name 'chorma_db'

DB_PATH = str(Path(__file__).resolve().parent.parent / "chorma_db")
client = chromadb.PersistentClient(path = DB_PATH)

#creaying collection

collection = client.get_or_create_collection(name="askmydocs_collection")

def store_chunks(chunks:list[dict]):
    """
    Stores chunks(text, metadat , embeddings) into chromaDB
    """

    if not chunks:
        print("No chunks to store")
        return

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for chunk in chunks:

        #create unique id for each chunk  elimantining chance of duplications

        chunk_id = f"{chunk['filename']}_chunk{chunk['chunk_index']}"

        if chunk['embedding'] is None:
            continue

        ids.append(chunk_id)
        embeddings.append(chunk['embedding'])
        documents.append(chunk['text'])
        metadatas.append({
            'filename':chunk['filename'],
            'chunk_index': chunk['chunk_index'],
            'token_count':chunk['token_count']
        })
        
        # 'Upsert' means it will insert the chunks, or update them if the ID already exists

    collection.upsert(
        ids = ids,
        embeddings = embeddings,
        documents = documents,
        metadatas = metadatas
        )

    print(f"Successfully stored {len(ids)} chunks in ChromaDB")


def query_store(query_embedding: list[float], top_k: int =5) ->list[dict]:
    """
    Queries ChromaDB for the most similar chunks to the query embedding.
    
    """

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )

    # Chroma returns a "list of lists" to  query multiple questions at once.
    #  since one question at a time being asked , grabbing the index[0].

    retrieved_chunks = []

    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            retrieved_chunks.append({
                'id':results['ids'][0][i],
                'text':results['documents'][0][i],
                'metadata':results['metadatas'][0][i],
                'similarity_distance':results['distances'][0][i] if 'distances' in results else None 
            })
            
    return retrieved_chunks

#-----Simple Testing Block-----
if __name__ == "__main__":
    count = collection.count()
    print(f"ChromaDB initialized! Collection currently holds {count} chunks")