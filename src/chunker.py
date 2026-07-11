import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(raw_text: str, filename: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:

    """
    Splits raw text into overlappinf chunk based on token count.
    Returns a list of dictionaries each containg:
    'file_name',
    'chunk_index',
    'text_chunk'
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    if chunk_size <=0:
        raise ValueError("chunk_size must be positive")
    # intialize tokenizer

    encoder = tiktoken.get_encoding("cl100k_base")

    # 1. Convert entiere raw text into a list of token integers
    tokens = encoder.encode(raw_text)

    chunks = []
    chunk_index = 0
    
    #if the document is extremly short, it returns a singel chunk
    
    if len(tokens) <= chunk_size:
        chunks.append({
            "filename": filename,
            "chunk_index": chunk_index,
            "text": raw_text,
            "token_count": len(tokens)
        })
        return chunks
    # 2. slice the token list into overlapping windows
    start = 0
    while start < len(tokens):
        end = start+chunk_size
        chunk_tokens = tokens[start:end]

        #3. Decode the token to readble string

        chunk_str = encoder.decode(chunk_tokens)

        chunks.append({
            "filename":filename,
            "chunk_index":chunk_index,
            "text": chunk_str,
            "token_count": len(chunk_tokens)
        })

        chunk_index += 1

        # Advamce the window, but step back by overlap amount

        start += (chunk_size - chunk_overlap)

    return chunks