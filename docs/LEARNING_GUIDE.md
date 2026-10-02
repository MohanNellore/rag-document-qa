# Learning Guide: understand this project before you put it on your resume

You must be able to explain every file here in an interview. Follow this order.

## 1. Core concepts (learn these first)
| Term | Plain meaning |
|---|---|
| **LLM** | A model that predicts text. It only knows what it was trained on. |
| **Hallucination** | The LLM confidently says something false. |
| **RAG** | Fetch relevant text first, then give it to the LLM so it answers from it. |
| **Chunking** | Cutting documents into small pieces so retrieval is precise. |
| **Embedding** | A list of numbers representing a text; similar texts get similar numbers. |
| **Cosine similarity** | Measures the angle between two vectors (1 = same direction/meaning, 0 = unrelated). |
| **Top-k retrieval** | Take the k most similar chunks. |
| **TF-IDF** | Scores words by how often they appear in a chunk vs. across all chunks (keyword matching). |
| **Vector database** | Storage built for nearest-neighbour search (FAISS, Chroma, Pinecone). |

## 2. Read the code in this order
1. `src/rag/chunker.py`: the simplest file. Try `chunk_text` in a Python shell with different sizes.
2. `src/rag/embeddings.py`: see how text becomes vectors. Print `embedder.embed(["hello world"])`.
3. `src/rag/vector_store.py`: the heart of retrieval. Understand `scores = self.vectors @ q`.
4. `src/rag/llm.py`: see how the prompt is built from the retrieved chunks.
5. `src/rag/pipeline.py`: how all the steps connect and where the `min_score` guardrail sits.
6. `cli.py`, then `app.py`: the interfaces.
7. `evaluate.py` and `tests/`: how quality is measured.

## 3. Hands-on exercises (do these, they make the project yours)
1. **Experiment:** run `evaluate.py` with `--chunk-size 40`, `80`, `200` and note the Hit@k. Write the results table in your README.
2. **Compare embedders:** install `sentence-transformers` and compare against TF-IDF. Which does better on paraphrased questions such as "heart problems" vs "cardiac disease"?
3. **Add your own documents:** use your CPCB report or any PDFs, and write 10 new questions in `eval/eval_set.json`.
4. **Tune the guardrail:** change `min_score`. What happens to false refusals vs. wrong answers?
5. **Add a feature:** show the page number for PDF chunks, or add a "Clear index" button in the app.
6. **Swap the vector store:** replace the NumPy search with FAISS or Chroma and keep the same interface.
7. **Improve retrieval:** implement hybrid search (combine TF-IDF and embedding scores).

After each exercise, commit to Git with a meaningful message. A commit history showing your own work makes the repo credible.

## 4. Interview questions you should be able to answer
1. **What is RAG and why use it instead of fine-tuning?** It injects fresh or private knowledge at query time. It's cheaper, needs no retraining, can be updated by changing documents, and allows citations.
2. **How does your retrieval work?** Chunks and the query are embedded; cosine similarity ranks chunks; the top-k go into the prompt.
3. **How did you choose chunk size and overlap?** Explain your experiment results from exercise 1.
4. **TF-IDF vs. neural embeddings?** TF-IDF matches words, embeddings match meaning. Embeddings handle synonyms and paraphrases better.
5. **How do you reduce hallucination?** Grounded prompt, "say you don't know" instruction, similarity threshold, citations.
6. **How do you evaluate a RAG system?** Retrieval metrics (Hit@k, MRR) and answer metrics (faithfulness, relevance).
7. **What are the limitations of your project?** Brute-force search, naive chunking, no re-ranking, no conversation memory.
8. **How would you scale it?** Use a vector DB with approximate nearest-neighbour search, batch embedding, caching, async serving.
9. **What would you improve next?** Hybrid search, re-ranker, structure-aware chunking, evaluation of answer faithfulness.
10. **Why cosine similarity?** It compares direction, not length, so document length doesn't distort the score.

## 5. Resume bullets (use only after you've run and understood the project)
- Built an end-to-end Retrieval-Augmented Generation (RAG) application in Python with document ingestion, overlapping chunking, TF-IDF / Sentence-Transformer embeddings and cosine-similarity retrieval.
- Integrated LLM APIs (Gemini / OpenAI / Claude) with grounded prompts, source citations and a similarity-threshold guardrail to reduce hallucinated answers.
- Evaluated retrieval quality with a Hit@k benchmark across chunk sizes and embedding models; deployed a Streamlit interface and wrote unit tests.

Only claim results you have actually measured. If you tried chunk sizes, quote your own numbers.

## 6. Put it on GitHub
```bash
cd rag-document-qa
git init
git add .
git commit -m "Initial commit: RAG document Q&A pipeline"
git branch -M main
```
1. On github.com click **New repository**. Name it `rag-document-qa`, keep it Public, and do **not** tick "Add README".
2. Then run (replace `<your-username>`):
```bash
git remote add origin https://github.com/<your-username>/rag-document-qa.git
git push -u origin main
```
3. On the repo page: add a description ("RAG app for Q&A over your documents, built with Python"), add topics (`rag`, `llm`, `python`, `nlp`, `streamlit`, `generative-ai`), and add a screenshot of the Streamlit app to the README.
4. **Never commit your `.env` file or API keys.** `.gitignore` already blocks it.
5. Pin the repo on your GitHub profile and add the link to your resume header.

## 7. Free resources to learn more
- Anthropic, OpenAI and Google AI docs: search "retrieval augmented generation" for each.
- DeepLearning.AI short courses on RAG and LangChain (free).
- Hugging Face docs for Sentence-Transformers.
- FAISS and Chroma getting-started docs (for exercise 6).
- Stanford CS224N / Andrej Karpathy's YouTube videos on LLMs for background.
