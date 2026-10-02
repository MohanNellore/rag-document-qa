# 📄 RAG Document Q&A

A from-scratch **Retrieval-Augmented Generation (RAG)** application in Python. Ask questions about your own documents (`.txt`, `.md`, `.pdf`) and get answers **grounded in the retrieved sources**, instead of relying on what an LLM happened to memorise.

Built to understand every moving part of RAG, so the core pipeline uses plain **NumPy + scikit-learn** rather than hiding everything behind a framework.

## Features
- **Full RAG pipeline:** load → chunk → embed → store → retrieve → generate
- **Pluggable embeddings:** TF-IDF (offline, zero setup) or Sentence-Transformers (semantic)
- **Pluggable LLMs:** Gemini, OpenAI, Claude, or `none` (works with no API key)
- **Guardrail:** refuses to answer when retrieval confidence is too low (reduces hallucination)
- **Source citations** with similarity scores
- **Evaluation script** that measures retrieval quality (Hit@k) so you can run real experiments
- CLI + Streamlit web app + unit tests

## Architecture

```
 INDEXING                                         QUERYING
 ────────                                         ────────
 Documents ─► Chunker ─► Embedder ─► Vector Store ◄── Embedder ◄── Question
 (.txt/.pdf)  (overlap)  (TF-IDF /    (NumPy,           │
                          SBERT)      cosine sim)       ▼
                                          └─► top-k chunks ─► Prompt ─► LLM ─► Answer + Sources
```

## Project structure
```
rag-document-qa/
├── app.py                 # Streamlit web UI
├── cli.py                 # Command-line interface
├── evaluate.py            # Retrieval evaluation (Hit@k)
├── src/rag/
│   ├── loader.py          # read txt / md / pdf
│   ├── chunker.py         # overlapping word chunks
│   ├── embeddings.py      # TF-IDF and Sentence-Transformers
│   ├── vector_store.py    # NumPy vector DB with cosine search
│   ├── llm.py             # prompt + Gemini / OpenAI / Claude / offline
│   └── pipeline.py        # ties everything together
├── data/sample_docs/      # demo documents
├── eval/eval_set.json     # questions + expected source documents
├── tests/test_rag.py      # unit tests
└── docs/LEARNING_GUIDE.md # concepts, exercises, interview Q&A
```

## Quick start
```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**1. Works immediately, no API key:**
```bash
python cli.py ingest --docs data/sample_docs
python cli.py ask "What are the AQI categories in India?"
```

**2. With an LLM for natural-language answers** (Gemini has a free tier):
```bash
pip install google-genai
cp .env.example .env            # then add your GEMINI_API_KEY
python cli.py ask "Why is PM2.5 dangerous?" --provider gemini
```

**3. Web app:**
```bash
streamlit run app.py
```

**4. Tests and evaluation:**
```bash
python -m unittest discover -s tests -v
python evaluate.py
python evaluate.py --embedder sentence-transformers   # after: pip install sentence-transformers
```

## Design decisions
| Decision | Reason |
|---|---|
| Built the vector store with NumPy | Shows exactly how similarity search works; swap for FAISS/Chroma at scale |
| Overlapping chunks | Preserves context across chunk boundaries |
| Normalised vectors | Dot product equals cosine similarity, which keeps search fast |
| `min_score` threshold | Prevents confident answers from irrelevant context |
| Prompt says "answer ONLY from context" | Grounds the answer and enables citations |

## Limitations & future work
- Brute-force search is fine for thousands of chunks, not millions (use FAISS/Chroma).
- Word-based chunking ignores document structure (headings, tables).
- Add hybrid search (TF-IDF + embeddings) and a re-ranker.
- Add answer-quality evaluation (faithfulness), not just retrieval Hit@k.
- Add conversation memory for follow-up questions.

## License
MIT
