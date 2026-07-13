from rank_bm25 import BM25Okapi

class KeywordSearcher:

    def __init__(self):
        self.bm25 = None
        self.chunk_data = []

    def build_index(self,chunks: list[dict]):

        """
        Takes the started chunks and breaks them in to words (tokenization),
        then builds the BM25 index for fast keyword lookup.

        """
        self.chunk_data = chunks

        tokenized_corpus = []
        for chunk in chunks:
            words = chunk['text'].lower().split()
            tokenized_corpus.append(words)

        # Building BM25 Index
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 keyword Index built with {len(chunks)} chunks.")
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Searches the Bm25 index for the exact keywords in query.

        """

        if not self.bm25:
            print("Warning: keyword index not built yet!")

        tokenized_query = query.lower().split()

        # Retriving top N matching chunks
        top_chunks = self.bm25.get_top_n(tokenized_query,self.chunk_data,n=top_k)
        return top_chunks
    
if __name__ == "__main__":
    # --- SIMPLE TEST ---
    fake_chunks = [
        {"text": "The patient was diagnosed with severe Cardiomegaly.", "metadata": {"chunk_index": 1}},
        {"text": "XEB-FuseNet is an AI architecture for Thoracic diseases.", "metadata": {"chunk_index": 2}},
        {"text": "Apples are delicious and good for your health.", "metadata": {"chunk_index": 3}}
    ]
    
    searcher = KeywordSearcher()
    searcher.build_index(fake_chunks)
    
    print("\n--- Testing BM25 Keyword Search ---")
    
    # Notice we ask for an exact acronym
    results = searcher.search("What is XEB-FuseNet?", top_k=1)
    
    print("\nTop Result Found:")
    print(results[0]["text"])


