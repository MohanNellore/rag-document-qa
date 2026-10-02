"""Step 3 - EMBED: turn text into vectors so we can measure similarity.

Two interchangeable embedders:
  * TfidfEmbedder              - keyword based, works offline, no downloads.
  * SentenceTransformerEmbedder - semantic (meaning based) neural embeddings.

Both return L2-normalised vectors, so a dot product == cosine similarity.
"""
import numpy as np


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype("float32")


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:  # pre-trained: nothing to fit
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")


def get_embedder(name: str = "tfidf"):
    if name == "tfidf":
        return TfidfEmbedder()
    if name in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder '{name}'. Use 'tfidf' or 'sentence-transformers'.")
