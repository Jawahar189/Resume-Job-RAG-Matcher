"""Thin wrapper around an in-memory ChromaDB collection holding resume chunks."""

import hashlib
from dataclasses import dataclass

import chromadb
import numpy as np

from .chunking import Chunk


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    section: str
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.EphemeralClient()

    def create_collection(self, name_seed: str, chunks: list[Chunk], embeddings: np.ndarray) -> "Collection":
        name = "resume-" + hashlib.sha1(name_seed.encode("utf-8")).hexdigest()[:16]
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # we pass our own embeddings
        )
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{"section": c.section} for c in chunks],
        )
        return Collection(collection)


class Collection:
    def __init__(self, collection) -> None:
        self._collection = collection

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        top_k = min(top_k, self._collection.count())
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchHit(chunk_id=cid, text=doc, section=meta["section"], score=1.0 - dist)
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
