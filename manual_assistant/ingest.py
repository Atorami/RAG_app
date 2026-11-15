"""Load manuals, split them into chunks and scan every chunk before indexing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .guard import Finding, normalize, scan_text

SUPPORTED = {".md", ".txt", ".pdf"}
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    source: str            # file name
    location: str          # section heading or "page N"
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def quarantined(self) -> bool:
        return bool(self.findings)

    def label(self) -> str:
        return f"{self.source} — {self.location}" if self.location else self.source


def load_documents(folder: Path) -> list[tuple[str, str, str]]:
    """Return (file name, location, text) sections for every supported file."""
    if not folder.exists():
        raise FileNotFoundError(f"Documents folder not found: {folder}")
    sections: list[tuple[str, str, str]] = []
    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED):
        name = str(path.relative_to(folder))
        if path.suffix.lower() == ".pdf":
            sections.extend((name, loc, text) for loc, text in _read_pdf(path))
        else:
            sections.extend((name, loc, text) for loc, text in _split_markdown(path.read_text("utf-8")))
    return sections


def _read_pdf(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader  # imported lazily: only needed for PDFs

    reader = PdfReader(str(path))
    return [
        (f"page {i}", page.extract_text() or "")
        for i, page in enumerate(reader.pages, start=1)
    ]


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """Split on headings so each chunk can be cited by its section title."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections.append((m.group(2), body))
    return sections


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries where possible, keeping chunks under `size` chars."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:  # one huge paragraph: hard-split it
            head, para = para[:size], para[size - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(sections: list[tuple[str, str, str]], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, location, text in sections:
        for piece in split_text(text, size, overlap):
            digest = hashlib.sha1(f"{source}|{location}|{piece}".encode()).hexdigest()[:12]
            findings = scan_text(piece)
            chunks.append(
                Chunk(
                    id=digest,
                    source=source,
                    location=location,
                    # Hidden characters are stripped from what gets indexed and shown to the model.
                    text=normalize(piece),
                    findings=findings,
                )
            )
    return chunks
