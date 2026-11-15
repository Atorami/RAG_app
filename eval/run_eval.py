"""Score the assistant against eval/questions.yaml.

    python eval/run_eval.py                 # retrieval only, no model needed
    python eval/run_eval.py --llm           # full answers via the configured model

Uses the index and settings from the environment (MA_INDEX_PATH, MA_EMBEDDER, ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manual_assistant.config import get_settings  # noqa: E402
from manual_assistant.embeddings import make_embedder  # noqa: E402
from manual_assistant.llm import OpenAICompatibleChat  # noqa: E402
from manual_assistant.pipeline import ManualAssistant  # noqa: E402
from manual_assistant.store import VectorIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also generate and check answers")
    parser.add_argument("--questions", default=str(Path(__file__).with_name("questions.yaml")))
    args = parser.parse_args()

    s = get_settings()
    index = VectorIndex.load(s.index_path)
    embedder = make_embedder(s.embedder, s.embedding_model)
    cases = yaml.safe_load(Path(args.questions).read_text("utf-8"))

    passed = 0
    for case in cases:
        q, expect = case["question"], case["expect_source"]
        hits = [h for h in index.search(embedder.embed([q])[0], s.top_k) if h.score >= s.min_score]
        files = [h.chunk.source for h in hits]
        ok = (not hits) if expect == "none" else expect in files
        detail = f"retrieved {files or 'nothing'}"

        if ok and args.llm:
            llm = OpenAICompatibleChat(s.llm_base_url, s.llm_model, s.llm_api_key, s.llm_timeout_s, s.llm_max_tokens)
            answer = ManualAssistant(
                index, embedder, llm, top_k=s.top_k, min_score=s.min_score,
                allowed_link_domains=s.allowed_link_domains,
            ).ask(q)
            if expect == "none":
                ok = answer.status == "not_found"
            else:
                missing = [t for t in case["expect_terms"] if t.lower() not in answer.text.lower()]
                ok = answer.status == "answered" and not missing
                detail = f"status={answer.status} missing={missing}"

        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {q}  ({detail})")

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
