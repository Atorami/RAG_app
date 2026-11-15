"""Prompt-injection defences: scanning untrusted text and checking model output.

These checks are one layer, not a wall. Pattern matching is easy to bypass
(paraphrase, another language, encoding), which is why the pipeline also
marks retrieved text as data, verifies citations and inspects the answer.
See SECURITY.md for the full threat model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

# Phrases that try to override instructions, change the model's role or open a
# fake conversation turn. English, Polish and Russian, since manuals and
# attackers in this domain use all three.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    ("override", r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,20}\b(instruction|instructions|rules|prompt|context)"),
    ("override", r"\bnew (instructions|rules)\b\s*[:\-]"),
    ("role", r"\byou are now\b|\bact as\b.{0,30}\b(admin|developer|system|unrestricted)"),
    ("role", r"(^|\n)\s*(system|assistant|developer)\s*:"),
    ("role", r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?INST\]|###\s*(system|instruction)"),
    ("prompt_leak", r"\b(reveal|print|show|repeat|output)\b.{0,30}\b(system prompt|hidden instructions|your instructions)"),
    ("exfiltration", r"!\[[^\]]*\]\(https?://|\b(send|post|upload|forward)\b.{0,40}\b(to|at)\b.{0,10}(https?://|\S+@\S+)"),
    # Polish
    ("override", r"\b(zignoruj|pomiń|zapomnij)\b.{0,40}\b(poprzednie|wcześniejsze|wszystkie)\b.{0,20}\b(instrukcje|polecenia|zasady)"),
    # Russian
    ("override", r"(игнорируй|проигнорируй|забудь|отмени)\b.{0,40}(предыдущие|прошлые|все|системные)\b.{0,20}(инструкции|указания|правила)"),
    ("role", r"\bты теперь\b"),
]
_COMPILED = [(kind, re.compile(p, re.IGNORECASE | re.DOTALL)) for kind, p in _INJECTION_PATTERNS]

# Characters used to hide text from a human reviewer while the model still reads it.
_HIDDEN_CHARS = {
    "​", "‌", "‍", "⁠", "﻿",  # zero-width
    "‪", "‫", "‬", "‭", "‮",  # bidi embedding / override
    "⁦", "⁧", "⁨", "⁩",            # bidi isolates
}
_TAG_CHARS = range(0xE0000, 0xE0080)  # Unicode "tag" block, invisible ASCII smuggling

_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str


def normalize(text: str) -> str:
    """NFKC-normalise and drop hidden characters, so look-alike tricks match patterns."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in text if ch not in _HIDDEN_CHARS and ord(ch) not in _TAG_CHARS
    )


def scan_text(text: str) -> list[Finding]:
    """Return injection indicators found in untrusted text (documents or questions)."""
    findings: list[Finding] = []
    hidden = sum(1 for ch in text if ch in _HIDDEN_CHARS or ord(ch) in _TAG_CHARS)
    if hidden:
        findings.append(Finding("hidden_characters", f"{hidden} invisible character(s)"))
    clean = normalize(text)
    for kind, pattern in _COMPILED:
        match = pattern.search(clean)
        if match:
            findings.append(Finding(kind, match.group(0)[:80]))
    return findings


@dataclass(frozen=True)
class OutputCheck:
    ok: bool
    reasons: list[str]


def check_answer(
    answer: str,
    *,
    source_texts: list[str],
    n_sources: int,
    canary: str,
    allowed_domains: frozenset[str] = frozenset(),
) -> OutputCheck:
    """Inspect a model answer before it reaches the user.

    - The system-prompt canary must not appear (OWASP LLM07, prompt leakage).
    - Every citation [n] must point to a source that was actually provided.
    - Links and e-mail addresses are allowed only for domains on an allowlist.
      Checking them against the sources is not enough: a poisoned source can
      carry the attacker's link itself.
    - Phone numbers must already exist in the sources.
    - A non-refusal answer must cite at least one source (OWASP LLM09).
    """
    reasons: list[str] = []
    if canary and canary in answer:
        reasons.append("system prompt leaked")

    cited = {int(n) for n in _CITATION_RE.findall(answer)}
    bad = sorted(n for n in cited if n < 1 or n > n_sources)
    if bad:
        reasons.append(f"cites non-existent source(s) {bad}")

    for url in _URL_RE.findall(answer):
        if not _domain_allowed(urlsplit(url).hostname or "", allowed_domains):
            reasons.append(f"link to a domain that is not allowlisted: {url}")
    for email in _EMAIL_RE.findall(answer):
        if not _domain_allowed(email.rsplit("@", 1)[-1], allowed_domains):
            reasons.append(f"e-mail to a domain that is not allowlisted: {email}")
    corpus = _squash(" ".join(source_texts))
    for phone in _PHONE_RE.findall(answer):
        if _squash(phone) not in corpus:
            reasons.append(f"phone number not present in sources: {phone.strip()}")

    if not cited and not is_refusal(answer):
        reasons.append("answer has no citation")

    for finding in scan_text(answer):
        if finding.kind in {"role", "override", "exfiltration"}:
            reasons.append(f"answer contains injection marker ({finding.kind})")

    return OutputCheck(ok=not reasons, reasons=reasons)


REFUSAL = "I can't find this in the provided documents."


def is_refusal(answer: str) -> bool:
    """True only when the whole answer is the refusal sentence (quoting it inside a longer reply does not count)."""
    return answer.strip().strip('"').rstrip(".").lower() == REFUSAL.rstrip(".").lower()


def _domain_allowed(host: str, allowed: frozenset[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in allowed)


def _squash(text: str) -> str:
    return re.sub(r"[\s\-()]", "", text).lower()
