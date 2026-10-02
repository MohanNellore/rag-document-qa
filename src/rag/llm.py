"""Step 5 - GENERATE: give the LLM the retrieved context and ask it to answer.

The key idea of RAG: we DON'T rely on what the model memorised. We paste the
relevant chunks into the prompt and tell it to answer ONLY from them. This
reduces hallucination and lets the answer cite its sources.

Providers: "none" (no API key needed), "gemini", "openai", "anthropic".
Model names change often, so they are read from environment variables.
"""
import os

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Be concise and mention the source file names you used in square brackets, e.g. [file.txt]."
)


def build_prompt(question: str, hits: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in hits)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def generate_answer(question: str, hits: list[tuple[dict, float]], provider: str = "none") -> str:
    prompt = build_prompt(question, hits)
    if provider == "none":
        return _extractive(hits)
    if provider == "gemini":
        from google import genai  # pip install google-genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        resp = client.models.generate_content(model=model, contents=f"{SYSTEM_PROMPT}\n\n{prompt}")
        return resp.text
    if provider == "openai":
        from openai import OpenAI  # pip install openai

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    if provider == "anthropic":
        import anthropic  # pip install anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    raise ValueError("provider must be one of: none, gemini, openai, anthropic")


def _extractive(hits: list[tuple[dict, float]]) -> str:
    """Offline fallback: return the best matching passage (no LLM involved)."""
    chunk, _ = hits[0]
    return f"(No LLM configured - showing the most relevant passage)\n{chunk['text']} [{chunk['source']}]"
