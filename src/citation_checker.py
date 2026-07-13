import re

def check_citation(answer: str, retrieved_chunks: list[dict]) -> bool:
    """
    Takes the model's generated answer and retrieved chunks.
    Returns True if citations are valid, False if it caught a hallucination.
    """
    # Tolerant pattern: captures everything between "Chunk" and the closing ")",
    # regardless of how the model separates multiple chunk numbers
    # (e.g. "Chunk 23, 24", "Chunk 23, Chunk 24", "Chunk 23 and 24" all work).
    citation_pattern = r"\(Source:\s*(.*?),\s*Chunk\s*(.*?)\)"

    # Split the answer into segments, each ending right after a citation.
    # This lets us associate nearby numbers with the citation that follows them.
    segments = re.split(r"(\(Source:\s*.*?,\s*Chunk\s*.*?\))", answer)

    is_valid = True
    i = 0
    while i < len(segments) - 1:
        text_segment = segments[i]
        citation_match = re.match(citation_pattern, segments[i + 1])

        if citation_match:
            filename, chunk_numbers_raw = citation_match.groups()

            # Pull out every number regardless of separator/wording in between
            chunk_indices = [int(n) for n in re.findall(r"\d+", chunk_numbers_raw)]

            local_numbers = set(re.findall(r"\d+\.?\d*", text_segment))

            matching_texts = []
            for chunk in retrieved_chunks:
                if chunk['metadata']['filename'] == filename and chunk['metadata']['chunk_index'] in chunk_indices:
                    matching_texts.append(chunk['text'])

            if not matching_texts:
                print(f"Warning! Model cited chunks that don't exist! {filename}, Chunks {chunk_indices}")
                is_valid = False
            else:
                combined_text = " ".join(matching_texts)
                for num in local_numbers:
                    if num not in combined_text:
                        print(f"Hallucination Detected! Claim near citation to {filename} Chunks {chunk_indices} mentions '{num}', not found in those chunks.")
                        is_valid = False
        i += 2

    return is_valid