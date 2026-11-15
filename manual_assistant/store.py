"""A small vector index persisted to one .npz file.

Quarantined chunks are stored too, so they can be reviewed, but they are
never returned by search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .guard import Finding
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder_name = embedder_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        vectors = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
        return cls(chunks, vectors, embedder.name)

    @property
    def quarantined(self) -> list[Chunk]:
        return [c for c in self.chunks if c.quarantined]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Hit]:
        allowed = [i for i, c in enumerate(self.chunks) if not c.quarantined]
        if not allowed:
            return []
        scores = self.vectors[allowed] @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[allowed[i]], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = [
            {
                "id": c.id,
                "source": c.source,
                "location": c.location,
                "text": c.text,
                "findings": [[f.kind, f.detail] for f in c.findings],
            }
            for c in self.chunks
        ]
        # Plain JSON + float arrays only: no pickle, so loading an index cannot run code.
        np.savez_compressed(
            path,
            vectors=self.vectors,
            meta=np.array(json.dumps(meta)),
            embedder=np.array(self.embedder_name),
        )

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            chunks = [
                Chunk(
                    id=m["id"],
                    source=m["source"],
                    location=m["location"],
                    text=m["text"],
                    findings=[Finding(k, d) for k, d in m["findings"]],
                )
                for m in meta
            ]
            return cls(chunks, data["vectors"], str(data["embedder"]))
