"""Step 1 - LOAD: read raw documents (.txt, .md, .pdf) from a file or folder."""
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # imported lazily so .txt-only users don't need it

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(path: str) -> list[dict]:
    """Return a list of {"source": filename, "text": full text}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED
    )
    docs = []
    for f in files:
        text = _read_file(f)
        if text.strip():
            docs.append({"source": f.name, "text": text})
    if not docs:
        raise ValueError(f"No readable documents found in: {path}")
    return docs
