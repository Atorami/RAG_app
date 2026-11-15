"""Retrieve → build a guarded prompt → generate → check the answer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .embeddings import Embedder
from .guard import REFUSAL, check_answer, is_refusal, scan_text
from .llm import ChatModel
from .prompt import build_system_prompt, build_user_prompt, new_canary
from .store import VectorIndex

log = logging.getLogger(__name__)


@dataclass
class Source:
    id: int
    file: str
    location: str
    score: float


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    status: str = "answered"          # answered | not_found | blocked
    warnings: list[str] = field(default_factory=list)


class ManualAssistant:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        llm: ChatModel,
        *,
        top_k: int = 4,
        min_score: float = 0.25,
        check_output: bool = True,
        allowed_link_domains: frozenset[str] = frozenset(),
    ) -> None:
        if index.embedder_name != embedder.name:
            raise ValueError(
                f"Index was built with {index.embedder_name!r} but the embedder is {embedder.name!r}; re-run ingest."
            )
        self.index = index
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.check_output = check_output
        self.allowed_link_domains = allowed_link_domains

    def ask(self, question: str) -> Answer:
        warnings = [f"question: {f.kind}" for f in scan_text(question)]
        if warnings:
            # The asker is a trusted user, but log it: it may be a test or a pasted document.
            log.warning("Injection markers in question: %s", warnings)

        query_vector = self.embedder.embed([question])[0]
        hits = self.index.search(query_vector, self.top_k)
        hits = [h for h in hits if h.score >= self.min_score]
        if not hits:
            return Answer(REFUSAL, status="not_found", warnings=warnings)

        sources = [
            Source(i, h.chunk.source, h.chunk.location, round(h.score, 3))
            for i, h in enumerate(hits, start=1)
        ]
        canary = new_canary()
        raw = self.llm.complete(build_system_prompt(canary), build_user_prompt(question, hits))

        if is_refusal(raw):
            return Answer(REFUSAL, status="not_found", warnings=warnings)

        if self.check_output:
            check = check_answer(
                raw,
                source_texts=[h.chunk.text for h in hits],
                n_sources=len(hits),
                canary=canary,
                allowed_domains=self.allowed_link_domains,
            )
            if not check.ok:
                log.warning("Answer blocked: %s", check.reasons)
                return Answer(
                    "The answer was withheld because it failed a safety check. "
                    "Check the cited manual sections directly.",
                    sources=sources,
                    status="blocked",
                    warnings=warnings + check.reasons,
                )

        return Answer(raw, sources=sources, warnings=warnings)
