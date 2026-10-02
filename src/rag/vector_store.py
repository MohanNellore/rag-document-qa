"""Step 4 - STORE & SEARCH: a tiny vector database built with NumPy.

Production systems use FAISS / Chroma / Pinecone, but the core idea is exactly
this: keep a matrix of chunk vectors and find the ones closest to the query.
"""
import json
from pathlib import Path

import numpy as np

from .embeddings import get_embedder


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[dict] = []
        self.vectors: np.ndarray | None = None

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed(texts)

    def search(self, query: str, k: int = 4) -> list[tuple[dict, float]]:
        """Return the top-k (chunk, cosine_score) pairs for a query."""
        if self.vectors is None:
            raise RuntimeError("Index is empty. Build or load an index first.")
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q  # cosine similarity (vectors are normalised)
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ---- persistence (plain JSON, easy to inspect) ----
    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.json").write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        d = Path(directory)
        if not (d / "chunks.json").exists():
            raise FileNotFoundError(f"No index found in '{directory}'. Run ingest first.")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        store = cls(get_embedder(meta["embedder"]))
        store.build(json.loads((d / "chunks.json").read_text(encoding="utf-8")))
        return store
