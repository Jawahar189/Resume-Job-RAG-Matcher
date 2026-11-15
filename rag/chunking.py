"""Section-aware chunking.

Resumes are short and structured, so instead of fixed-size windows we split on
section headings (Experience, Skills, ...) and group consecutive lines into
small chunks. Small chunks give sharper embeddings and more precise evidence.
"""

import re
from dataclasses import dataclass

SECTION_KEYWORDS = (
    "summary", "profile", "objective", "about", "experience", "employment", "work history",
    "projects", "skills", "technical skills", "education", "certifications", "certificates",
    "achievements", "awards", "publications", "languages", "interests", "volunteer",
    "responsibilities", "requirements", "qualifications", "nice to have", "preferred",
    "what you will do", "what we offer", "about the role", "about you", "benefits",
)

_BULLET = re.compile(r"^\s*[-*\u2022\u25cf\u25aa\u2013]\s*")


@dataclass
class Chunk:
    id: str
    text: str
    section: str


def _is_heading(line: str) -> bool:
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped.split()) > 5:
        return False
    lowered = stripped.lower()
    if any(lowered == k or lowered.startswith(k) for k in SECTION_KEYWORDS):
        return True
    return stripped.isupper() and len(stripped) > 3


def chunk_document(text: str, max_words: int = 60, prefix: str = "chunk") -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "General"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            body = " ".join(buffer)
            chunks.append(Chunk(id=f"{prefix}-{len(chunks)}", text=f"{section}: {body}", section=section))
            buffer.clear()

    for raw in text.splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line:
            continue
        if _is_heading(line):
            flush()
            section = line.rstrip(":").strip().title()
            continue
        if buffer and len(" ".join(buffer).split()) + len(line.split()) > max_words:
            flush()
        buffer.append(line)
    flush()
    return chunks
