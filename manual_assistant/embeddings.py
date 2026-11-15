"""Text embedders behind one small interface."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised vectors."""


class HashingEmbedder:
    """Bag-of-words embedder with no model download.

    Good enough for tests, CI and offline demos on keyword-heavy manuals
    (alarm codes, part numbers). Use sentence-transformers for real use.
    """

    name = "hashing"

    _STOP = frozenset(
        "a an and are as at be by can do does for from has have how i if in is it its of on "
        "or should that the this to was what when which who why will with you your my".split()
    )

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        tokens = []
        for word in re.findall(r"[\w-]+", text.lower()):
            if word in self._STOP:
                continue
            tokens.append(word)
            if "-" in word:  # "f-0021" also matches "0021"; "rs-485" matches "485"
                tokens.extend(part for part in word.split("-") if part)
        return tokens

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _normalize(np.log1p(out))


class SentenceTransformerEmbedder:
    """Local neural embeddings (downloads the model on first use)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vectors.astype(np.float32))


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder: {kind}")


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
