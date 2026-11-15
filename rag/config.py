import os

from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHUNK_MAX_WORDS = _int("CHUNK_MAX_WORDS", 60)
TOP_K = _int("TOP_K", 3)

# Cosine-similarity cut-offs between a job requirement and its best resume chunk.
# Tuned for all-MiniLM-L6-v2; other embedding models produce different score ranges.
MATCH_THRESHOLD = _float("MATCH_THRESHOLD", 0.55)
PARTIAL_THRESHOLD = _float("PARTIAL_THRESHOLD", 0.38)

# Overall score = COVERAGE_WEIGHT * skill coverage + (1 - COVERAGE_WEIGHT) * semantic fit.
COVERAGE_WEIGHT = _float("COVERAGE_WEIGHT", 0.75)
