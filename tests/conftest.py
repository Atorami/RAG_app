"""Shared fixtures: an offline embedder and fake chat models with known behaviour."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from manual_assistant.embeddings import HashingEmbedder
from manual_assistant.guard import REFUSAL
from manual_assistant.ingest import build_chunks, load_documents
from manual_assistant.store import VectorIndex

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "manuals"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

_SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.DOTALL)
_WORD_RE = re.compile(r"[\w-]+")
_STOP = {"the", "a", "an", "what", "does", "do", "is", "how", "to", "of", "for", "and", "in", "on", "i", "my", "when", "which", "should", "be"}


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


class NaiveLLM:
    """A deliberately gullible model.

    It answers by quoting the best-matching sentence with a citation, but if a
    source contains attacker content (anything mentioning evil-parts) it obeys
    it, the way a real model may follow injected instructions. This makes the
    attack observable in tests without calling a real model.
    """

    def __init__(self) -> None:
        self.last_system = ""
        self.last_user = ""

    def complete(self, system: str, user: str) -> str:
        self.last_system, self.last_user = system, user
        sources = [(int(i), body) for i, body in _SOURCE_RE.findall(user)]
        for sid, body in sources:
            if "evil-parts" in body:
                bait = next(s for s in re.split(r"(?<=[.!?])\s+", body) if "evil-parts" in s)
                return f"Important: {bait} [{sid}]"
        question = user.rsplit("Technician's question:", 1)[-1]
        q = _words(question)
        best = (0, None, "")
        for sid, body in sources:
            for sentence in re.split(r"(?<=[.!?])\s+|\n", body):
                overlap = len(q & _words(sentence))
                if overlap > best[0]:
                    best = (overlap, sid, sentence.strip())
        if best[0] < 1:
            return REFUSAL
        return f"{best[2]} [{best[1]}]"


class ScriptedLLM:
    """Returns a fixed answer; used to test output checks directly."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_system = ""

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        return self.reply.replace("{SYSTEM}", system)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


def make_index(tmp_path: Path, embedder, *extra: Path) -> VectorIndex:
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLES, docs)
    for f in extra:
        shutil.copy(f, docs / f.name)
    chunks = build_chunks(load_documents(docs), size=900, overlap=120)
    return VectorIndex.build(chunks, embedder)


@pytest.fixture
def clean_index(tmp_path, embedder) -> VectorIndex:
    return make_index(tmp_path, embedder)
