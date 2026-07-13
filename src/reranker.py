from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self , model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder model.
        The MS-MARCO model is specifically trianed on bing search data 
        To know exactly how relevent a paragrapg is to a user's question.

        """
        print(f"Loading self incoder model : {model_name}.. this might take few moment for the first time")
        self.model = CrossEncoder(model_name)


    def rerank(self,query: str, retrieved_chunks: list[dict]) -> list[dict]:
        """
        Takes the top Chunks from Hybrid Retriever, Scores them against the query.
        Returns them sorted by absoulate relevence.

        """

        if not retrieved_chunks:
            return []
        
        # Cross Encoder except inputs as list of pairs . Eg: [[query, text1], [query, text2], ...]

        pairs = []
        for chunk in retrieved_chunks:
            pairs.append([query,chunk["text"]])

        # Score the pairs - faster in local
        scores = self.model.predict(pairs)

        # Attching score to the chunk 

        for chunk , score in zip(retrieved_chunks, scores):
            chunk['rerank_score'] = float(score)

        # Sort chunks by rerank score
        reranked_chunks = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)

        return reranked_chunks
        
            