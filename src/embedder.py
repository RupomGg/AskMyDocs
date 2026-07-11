import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

#loading api key from .env file
load_dotenv()

#intialize gemini client once which autometcally look for api key inside enviorment file .

client =  genai.Client()

def embed_document_chunks(chunks: list[dict]) -> list[dict]:

    """
    takes a list of chunk dictonaries and calls gemini to get embeddings 
    adds the 'embedding' key to each dictonary , and  returns the list .
    """

    for chunk in chunks:
        try:
            # using RETRIEVAL_DOCUMENT for chunk going into database
            response = client.models.embed_content(
                model = 'gemini-embedding-001',
                contents = chunk["text"],
                config = types.EmbedContentConfig(task_type ="RETRIEVAL_DOCUMENT")
            )
            # The response contains a list of embeddings. We grab the first (and only) one.            
            chunk["embedding"] = response.embeddings[0].values
        except Exception as e:
            print(f"Error embedding chunk {chunk['chunk_index']} from {chunk['filename']}: {e}")
            chunk["embedding"] = None



    return chunks

def embed_query(question: str) -> list[float]:
    """
    takes a single user question and returns its embedding vector.

    """
    try:
        # Using RETRIEVAL_QUERY for user's Questions
        response = client.models.embed_content(
            model = 'gemini-embedding-001',
            contents = question,
            config = types.EmbedContentConfig(task_type = "RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []


if __name__ == "__main__":
    print("Testing embed_query...")
    query_vector = embed_query("What programming languages does this person know?")
    
    if query_vector:
        print(f"Success! Query vector generated.")
        print(f"Length of the vector (dimensions): {len(query_vector)}") # Should be 3072
        print(f"First 5 values: {query_vector[:5]}")
    else:
        print("Failed to generate query vector. Check your API key!")



            

