"""Local embeddings (no API key needed).

The default model, all-MiniLM-L6-v2, runs through ChromaDB's bundled ONNX build, so
PyTorch is not required. Any other EMBEDDING_MODEL is loaded with sentence-transformers.
"""

import numpy as np

from . import config

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        if model_name.split("/")[-1] == DEFAULT_MODEL:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            onnx_model = ONNXMiniLM_L6_V2()
            self._encode = lambda texts: np.asarray(onnx_model(texts), dtype=np.float32)
        else:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model_name)
            self._encode = lambda texts: st_model.encode(texts, convert_to_numpy=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized vectors, so a dot product equals cosine similarity."""
        vectors = self._encode(texts)
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Average of chunk vectors: a whole-document embedding that is not truncated
    by the model's max sequence length."""
    mean = vectors.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-10)
