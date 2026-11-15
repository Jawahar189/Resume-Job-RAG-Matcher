"""Extract plain text from uploaded PDF, DOCX or TXT files."""

import io
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


def load_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        text = "\n".join(parts)
    elif ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of {SUPPORTED_EXTENSIONS}.")
    return clean_text(text)


def load_file(path: str | Path) -> str:
    path = Path(path)
    return load_text(path.read_bytes(), path.name)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
