"""Settings read from environment variables, with safe local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Where documents and the index live. data/ is git-ignored on purpose:
    # manuals and customer documents never go into the repository.
    docs_dir: Path = Path(os.getenv("MA_DOCS_DIR", "data/manuals"))
    index_path: Path = Path(os.getenv("MA_INDEX_PATH", "data/index.npz"))

    # Chunking
    chunk_size: int = int(os.getenv("MA_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("MA_CHUNK_OVERLAP", "120"))

    # Retrieval
    top_k: int = int(os.getenv("MA_TOP_K", "4"))
    # Embeddings: "sentence-transformers" (real) or "hashing" (offline, for tests and demos).
    embedder: str = os.getenv("MA_EMBEDDER", "sentence-transformers")
    # Below this cosine similarity a chunk counts as "not in the documents".
    # Scores are not comparable across embedders, so each has its own default;
    # tune it with eval/run_eval.py on your own documents.
    min_score: float = float(
        os.getenv("MA_MIN_SCORE")
        or ("0.1" if os.getenv("MA_EMBEDDER", "sentence-transformers") == "hashing" else "0.3")
    )
    embedding_model: str = os.getenv("MA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LLM: any OpenAI-compatible chat endpoint. The default is a local Ollama server,
    # so documents do not leave the machine unless you point it elsewhere on purpose.
    llm_base_url: str = os.getenv("MA_LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = os.getenv("MA_LLM_MODEL", "llama3.1:8b")
    llm_api_key: str = os.getenv("MA_LLM_API_KEY", "")
    llm_timeout_s: float = float(os.getenv("MA_LLM_TIMEOUT_S", "60"))
    llm_max_tokens: int = int(os.getenv("MA_LLM_MAX_TOKENS", "500"))

    # Domains the assistant may link to or give e-mail addresses for, e.g.
    # "vendor.example,mycompany.pl". Empty means answers never contain links.
    allowed_link_domains: frozenset[str] = field(
        default_factory=lambda: frozenset(d.lower() for d in _env_list("MA_ALLOWED_LINK_DOMAINS", ""))
    )

    # API limits (OWASP LLM10: unbounded consumption)
    max_question_chars: int = int(os.getenv("MA_MAX_QUESTION_CHARS", "500"))
    rate_limit_per_minute: int = int(os.getenv("MA_RATE_LIMIT_PER_MINUTE", "20"))
    allowed_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "MA_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
        )
    )


def get_settings() -> Settings:
    return Settings()
