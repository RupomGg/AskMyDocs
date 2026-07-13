from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
from citation_checker import check_citation
from prompts import ANSWER_PROMPT_V1


load_dotenv()

client = genai.Client()

GENERATION_MODEL = "gemini-3-flash-preview"  # move to config.py once verified working


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Takes a question and list of retrieved chunks, builds a prompt,
    and calls Gemini to generate an answer with citations.
    """
    if not retrieved_chunks or len(retrieved_chunks) == 0:
        return "I could not find any relevant documents to answer your question."

    # 1. Build the context string from the retrieved chunks
    context_text = ""
    for idx, chunk in enumerate(retrieved_chunks):
        # explicitly label each chunk with its source file so the model knows what to cite
        context_text += f"\n---Source {idx + 1}: {chunk['metadata']['filename']} (Chunk {chunk['metadata']['chunk_index']})---\n"
        context_text += chunk['text'] + "\n"

    # 2. Build the prompt for the AI model
    prompt = ANSWER_PROMPT_V1.replace("{context}", context_text).replace("{question}", question)
    
    try:
        print("Generating your question's answer...")
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,  # low temperature keeps the AI factual
            )
        )
        final_answer = response.text

        # -- citation Validation enforcement --
        is_valid = check_citation(final_answer,retrieved_chunks)
        
        if not is_valid:
            final_answer += "\n\n CAUTION: The AI generated an answer, but one or more facts (like numbers) could not be verified against the exact cited source chunk. Please double check the document!"
        return final_answer

    except errors.APIError as e:
        if e.code == 429:
            print("\n ERROR: Rate limit or Quota Exceeded (429).")
            print("Model rate caps hit. Please wait a moment and try again.")
            return "Sorry, I hit a rate limit while generating the answer. Please wait a moment and try again."

        elif e.code in [403, 404]:
            print(f"\n ERROR: Model access issue ({e.code}).")
            print(f"'{GENERATION_MODEL}' might not be accessible on your current tier/region, or may be deprecated.")
            return f"Sorry, the model is currently unavailable (status {e.code}). It may be deprecated or access-restricted — check the model name."

        else:
            print(f"\n ERROR: API failed with status {e.code}: {e.message}")
            return f"Sorry, the request failed with an API error (status {e.code})."

    except Exception as e:
        if "timeout" in str(e).lower():
            print("\n ERROR: Network timeout while reaching model API.")
            return "Sorry, the request timed out while reaching the model API. Please try again."
        else:
            print(f"\n ERROR: Unexpected failure: {e}")
            return "Sorry, I encountered an unexpected error while trying to generate the answer."


# --- Simple Testing Block ---
if __name__ == "__main__":
    fake_chunks = [
        {
            "metadata": {"filename": "fake_doc.pdf", "chunk_index": 1},
            "text": "The XEB-FuseNet framework is designed for Thoracic Disease Diagnosis."
        }
    ]

    test_q = "What is XEB-FuseNet designed for?"
    print(f"Question: {test_q}\n")

    answer = generate_answer(test_q, fake_chunks)

    print("\n=== GEMINI'S ANSWER ===")
    print(answer)