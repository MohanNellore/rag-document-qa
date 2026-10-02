"""Streamlit web app.  Run with:  streamlit run app.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from rag import RAGPipeline  # noqa: E402

load_dotenv()
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A")
st.caption("Ask questions about your documents. Answers are grounded in retrieved sources.")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider", ["none", "gemini", "openai", "anthropic"],
                            help="'none' works offline and just shows the best matching passage.")
    embedder = st.selectbox("Embedder", ["tfidf", "sentence-transformers"])
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)
    uploads = st.file_uploader("Upload documents (.txt, .md, .pdf)", accept_multiple_files=True)
    use_samples = st.checkbox("Use sample documents", value=not uploads)
    build = st.button("Build index")

if build:
    with st.spinner("Indexing..."):
        rag = RAGPipeline(embedder=embedder, provider=provider, top_k=top_k)
        if uploads:
            tmp = Path(tempfile.mkdtemp())
            for f in uploads:
                (tmp / f.name).write_bytes(f.getbuffer())
            n = rag.ingest(str(tmp))
        elif use_samples:
            n = rag.ingest("data/sample_docs")
        else:
            st.error("Upload files or tick 'Use sample documents'.")
            st.stop()
        st.session_state["rag"] = rag
        st.success(f"Indexed {n} chunks.")

question = st.text_input("Your question")
if question:
    rag = st.session_state.get("rag")
    if rag is None:
        st.info("Click 'Build index' in the sidebar first.")
    else:
        rag.provider, rag.top_k = provider, top_k
        try:
            result = rag.ask(question)
        except Exception as e:  # e.g. missing API key
            st.error(f"Error: {e}")
            st.stop()
        st.subheader("Answer")
        st.write(result["answer"])
        if result["sources"]:
            st.subheader("Sources")
            for s in result["sources"]:
                with st.expander(f"{s['source']}  (similarity {s['score']})"):
                    st.write(s["text"])
