from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent # absolute path to the root directory

DOCUMENTS_DIR = BASE_DIR / "documents" # absolute path to the documents directory

CHUNK_SIZE = 600 # number of tokens per chunk

CHUNK_OVERLAP = 100 # number of tokens to overlap between chunks

