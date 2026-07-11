# Ask My Docs (RAG Pipeline)

A fully local Retrieval-Augmented Generation (RAG) pipeline built from scratch using Python, ChromaDB, and Google's Gemini models. It allows you to chat with your local PDF and text documents, generating AI answers backed by direct citations to your files.

## Features
- **Local Document Storage**: Safely processes and chunks PDFs/text files locally.
- **Fast Embeddings**: Uses `gemini-embedding-001` to vectorize text chunks.
- **Local Vector Database**: Persistently stores all document embeddings in a local `chroma_db` using ChromaDB.
- **Smart Retrieval**: Uses `gemini-3-flash-preview` to answer questions based *only* on the provided context, citing exact source files.
- **Interactive CLI**: Chat directly with your documents through a terminal interface.

## Prerequisites
1. Python 3.9+
2. A Google AI Studio API Key (Free tier works!)

## Setup

1. **Install dependencies**
   Make sure you have your virtual environment activated, then install the required packages:
   ```powershell
   pip install google-genai python-dotenv chromadb tiktoken pypdf markitdown
   ```

2. **Configure Environment Variables**
   Create a `.env` file in the root directory of the project and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_google_api_key_here
   ```

## Usage

1. **Add your documents**
   Place any `.txt` or `.pdf` files you want to chat with inside the `documents/` folder.

2. **Run the pipeline**
   Start the interactive chatbot by running the included PowerShell script:
   ```powershell
   .\run.ps1
   ```
   
   *This script will automatically create the `documents` folder if it doesn't exist, activate the virtual environment, and launch the pipeline.*

3. **Ask Questions!**
   Type your questions into the terminal. The AI will retrieve the most relevant chunks from your documents and generate a cited response. Type `exit` or `quit` to stop the program.

## Architecture
- **`loader.py`**: Ingests raw files from the `documents/` folder.
- **`chunker.py`**: Splits raw text into safe token-sized chunks with a 100-token overlap.
- **`embedder.py`**: Converts chunked text into 3072-dimensional vectors via Gemini.
- **`vector_store.py`**: Upserts and searches vectors in a local ChromaDB collection.
- **`retriever.py`**: Matches a user's question embedding against stored chunk embeddings.
- **`generator.py`**: Constructs a strict context prompt and generates the final cited answer.
- **`test_pipeline.py`**: The main execution loop that wires it all together.
