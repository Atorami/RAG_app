"""HTTP API and a minimal web page.

    uvicorn manual_assistant.api:app --port 8000
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .embeddings import make_embedder
from .llm import LLMUnavailable, OpenAICompatibleChat
from .pipeline import ManualAssistant
from .store import VectorIndex

settings = get_settings()
app = FastAPI(title="Manual Assistant", version="0.2.0")

# Only the origins you list may call the API from a browser (was "*" before).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_STATIC = Path(__file__).parent / "static"
_requests: dict[str, deque[float]] = defaultdict(deque)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=settings.max_question_chars)


@lru_cache(maxsize=1)
def get_assistant() -> ManualAssistant:
    index = VectorIndex.load(settings.index_path)
    embedder = make_embedder(settings.embedder, settings.embedding_model)
    llm = OpenAICompatibleChat(
        settings.llm_base_url, settings.llm_model, settings.llm_api_key,
        settings.llm_timeout_s, settings.llm_max_tokens,
    )
    return ManualAssistant(
        index, embedder, llm,
        top_k=settings.top_k,
        min_score=settings.min_score,
        allowed_link_domains=settings.allowed_link_domains,
    )


def _rate_limited(client: str) -> bool:
    now = time.monotonic()
    window = _requests[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return True
    window.append(now)
    return False


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question, request: Request) -> dict:
    client = request.client.host if request.client else "unknown"
    if _rate_limited(client):
        raise HTTPException(status_code=429, detail="Too many requests, try again in a minute.")
    try:
        return asdict(get_assistant().ask(body.question))
    except LLMUnavailable:
        raise HTTPException(status_code=503, detail="The language model is not reachable. Is Ollama running?")
