"""Command-line interface.

  python cli.py ingest --docs data/sample_docs
  python cli.py ask "What are the AQI categories?" --provider gemini
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag import RAGPipeline  # noqa: E402

try:  # optional: load API keys from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
INDEX_DIR = "index"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Document Q&A")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Build the vector index from documents")
    ing.add_argument("--docs", default="data/sample_docs")
    ing.add_argument("--embedder", default="tfidf", choices=["tfidf", "sentence-transformers"])

    ask = sub.add_parser("ask", help="Ask a question about the indexed documents")
    ask.add_argument("question")
    ask.add_argument("--provider", default="none", choices=["none", "gemini", "openai", "anthropic"])
    ask.add_argument("--top-k", type=int, default=4)

    args = parser.parse_args()

    if args.cmd == "ingest":
        n = RAGPipeline(embedder=args.embedder).ingest(args.docs, INDEX_DIR)
        print(f"Indexed {n} chunks from '{args.docs}' using {args.embedder}. Saved to ./{INDEX_DIR}/")
    else:
        rag = RAGPipeline(provider=args.provider, top_k=args.top_k)
        rag.load_index(INDEX_DIR)
        result = rag.ask(args.question)
        print("\nANSWER:\n" + result["answer"])
        if result["sources"]:
            print("\nSOURCES:")
            for s in result["sources"]:
                print(f"  - {s['source']} (similarity {s['score']})")


if __name__ == "__main__":
    main()
