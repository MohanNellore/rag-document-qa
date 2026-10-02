"""The full RAG pipeline: load -> chunk -> embed -> store -> retrieve -> generate."""
from .chunker import chunk_documents
from .embeddings import get_embedder
from .llm import generate_answer
from .loader import load_documents
from .vector_store import VectorStore

NO_ANSWER = "I couldn't find relevant information in the documents."


class RAGPipeline:
    def __init__(self, embedder: str = "tfidf", provider: str = "none",
                 chunk_size: int = 120, overlap: int = 30, top_k: int = 4, min_score: float = 0.05):
        self.embedder_name = embedder
        self.provider = provider
        self.chunk_size, self.overlap = chunk_size, overlap
        self.top_k, self.min_score = top_k, min_score
        self.store: VectorStore | None = None

    def ingest(self, path: str, index_dir: str | None = None) -> int:
        """Load + chunk + embed documents. Optionally save the index. Returns #chunks."""
        docs = load_documents(path)
        chunks = chunk_documents(docs, self.chunk_size, self.overlap)
        self.store = VectorStore(get_embedder(self.embedder_name))
        self.store.build(chunks)
        if index_dir:
            self.store.save(index_dir)
        return len(chunks)

    def load_index(self, index_dir: str) -> None:
        self.store = VectorStore.load(index_dir)

    def retrieve(self, question: str) -> list[tuple[dict, float]]:
        if self.store is None:
            raise RuntimeError("Call ingest() or load_index() first.")
        return self.store.search(question, self.top_k)

    def ask(self, question: str) -> dict:
        hits = self.retrieve(question)
        # Guardrail: if even the best chunk is a weak match, don't guess.
        if not hits or hits[0][1] < self.min_score:
            return {"answer": NO_ANSWER, "sources": []}
        answer = generate_answer(question, hits, self.provider)
        sources = [{"source": c["source"], "score": round(s, 3), "text": c["text"]} for c, s in hits]
        return {"answer": answer, "sources": sources}
