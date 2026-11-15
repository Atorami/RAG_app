"""Command line: ingest, ask, review quarantine.

    python -m manual_assistant ingest
    python -m manual_assistant ask "What does alarm F-0021 mean?"
    python -m manual_assistant quarantine
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .config import get_settings
from .embeddings import make_embedder
from .ingest import build_chunks, load_documents
from .llm import LLMUnavailable, OpenAICompatibleChat
from .pipeline import ManualAssistant
from .store import VectorIndex


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="manual_assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="index the documents folder")
    p_ingest.add_argument("--docs", default=str(settings.docs_dir))
    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    sub.add_parser("quarantine", help="list chunks excluded as possible injections")
    args = parser.parse_args(argv)

    embedder = make_embedder(settings.embedder, settings.embedding_model)

    if args.command == "ingest":
        from pathlib import Path

        sections = load_documents(Path(args.docs))
        chunks = build_chunks(sections, settings.chunk_size, settings.chunk_overlap)
        index = VectorIndex.build(chunks, embedder)
        index.save(settings.index_path)
        print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} file(s) → {settings.index_path}")
        if index.quarantined:
            print(f"Quarantined {len(index.quarantined)} chunk(s); review with: python -m manual_assistant quarantine")
        return 0

    index = VectorIndex.load(settings.index_path)

    if args.command == "quarantine":
        if not index.quarantined:
            print("No quarantined chunks.")
        for c in index.quarantined:
            print(f"- {c.label()}")
            for f in c.findings:
                print(f"    {f.kind}: {f.detail!r}")
        return 0

    llm = OpenAICompatibleChat(
        settings.llm_base_url, settings.llm_model, settings.llm_api_key,
        settings.llm_timeout_s, settings.llm_max_tokens,
    )
    assistant = ManualAssistant(
        index, embedder, llm,
        top_k=settings.top_k,
        min_score=settings.min_score,
        allowed_link_domains=settings.allowed_link_domains,
    )
    try:
        answer = assistant.ask(args.question)
    except LLMUnavailable as err:
        print(f"{err}\nStart a local model (e.g. `ollama run {settings.llm_model}`) or set MA_LLM_BASE_URL.", file=sys.stderr)
        return 2
    print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
