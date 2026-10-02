"""Measure RETRIEVAL quality: for each question, is the right document in the top-k?

  python evaluate.py
  python evaluate.py --embedder sentence-transformers

Run it for different embedders / chunk sizes and compare - that is an experiment!
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag import RAGPipeline  # noqa: E402


def evaluate(embedder: str = "tfidf", chunk_size: int = 120, overlap: int = 30, k: int = 3) -> float:
    rag = RAGPipeline(embedder=embedder, chunk_size=chunk_size, overlap=overlap, top_k=k)
    rag.ingest("data/sample_docs")
    items = json.loads(Path("eval/eval_set.json").read_text(encoding="utf-8"))
    hits = 0
    for item in items:
        retrieved = [c["source"] for c, _ in rag.retrieve(item["question"])]
        ok = item["expected_source"] in retrieved
        hits += ok
        print(f"{'PASS' if ok else 'FAIL'}  {item['question']}")
    score = hits / len(items)
    print(f"\nHit@{k} = {score:.2f}  (embedder={embedder}, chunk_size={chunk_size}, overlap={overlap})")
    return score


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embedder", default="tfidf")
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    evaluate(a.embedder, a.chunk_size, a.overlap, a.k)
