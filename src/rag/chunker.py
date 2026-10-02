"""Step 2 - CHUNK: split long documents into small overlapping pieces.

Why? An LLM has a limited context window, and retrieval works best when each
piece of text covers ONE idea. Overlap keeps sentences that fall on a boundary
from losing their context.
"""


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into chunks of `chunk_size` words with `overlap` shared words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int = 120, overlap: int = 30) -> list[dict]:
    """Chunk every document, keeping track of where each chunk came from."""
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})
    return chunks
