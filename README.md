# Manual Assistant

Ask questions about equipment manuals and get answers with a source for every fact. Built for technicians working on marine and industrial equipment, and hardened against **indirect prompt injection**: instructions hidden inside a document that try to hijack the model.

> Earlier versions of this repo were a generic RAG prototype (LangChain + Chroma + flan-t5 + a Node proxy). Version 0.2 is a rewrite focused on safe use with real technical documentation.

## What it does

```
manuals (PDF, MD, TXT) ──► split by section ──► scan for injection ──► vector index
                                                       │
                                               quarantine (excluded,
                                               listed for review)

question ──► retrieve top chunks ──► prompt with sources as tagged data ──► local LLM
                                                                              │
                     answer + sources ◄── output checks (citations, links, leaks)
```

- Answers only from the indexed documents and cites them as `[1]`, `[2]`, with file and section.
- Says *"I can't find this in the provided documents"* instead of guessing.
- Runs against a **local model by default** (Ollama), so manuals and customer documents do not leave the machine.

## Security design

Four layers, because each one alone can be bypassed. Details and the full threat model: [SECURITY.md](SECURITY.md).

| Layer | What it stops | Where |
| --- | --- | --- |
| Ingest scan and quarantine | Documents with obvious injection phrases (EN/PL/RU), fake chat turns, hidden Unicode | `guard.scan_text`, `ingest.build_chunks` |
| Sources as tagged data | Model treating document text as instructions; a document closing its own tag | `prompt.py` |
| Output checks | Links or e-mails outside an allowlist, invented phone numbers, citations to missing sources, answers without citations, system-prompt leaks (canary) | `guard.check_answer` |
| API limits | Oversized questions, request floods, cross-site calls (CORS allowlist), HTML injection in the UI (`textContent` only) | `api.py`, `static/index.html` |

The tests prove each layer with a deliberately gullible fake model that obeys injected instructions:

- **Without defences**, the poisoned bulletin makes the answer send the technician to `support@evil-parts.example`.
- **Layer 1** keeps that bulletin out of the prompt.
- **Layer 3** blocks a subtler planted note that passes the ingest scan: the link appears only in the model's answer, and it is blocked there.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings,dev]"

# 1. Put manuals in data/manuals/ (git-ignored), or try the fictional samples:
python -m manual_assistant ingest --docs samples/manuals
python -m manual_assistant quarantine          # review anything flagged

# 2. Start a local model and ask
ollama run llama3.1:8b                          # in another terminal
python -m manual_assistant ask "What does alarm F-0021 mean?"

# 3. Web UI on http://localhost:8000
uvicorn manual_assistant.api:app --port 8000
```

No GPU or model download needed for a quick look: set `MA_EMBEDDER=hashing` to use the offline keyword embedder.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MA_DOCS_DIR` | `data/manuals` | Folder with manuals |
| `MA_INDEX_PATH` | `data/index.npz` | Index file |
| `MA_EMBEDDER` | `sentence-transformers` | Or `hashing` (offline) |
| `MA_MIN_SCORE` | 0.3 (ST) / 0.1 (hashing) | Below this, the answer is "not found" |
| `MA_LLM_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `MA_LLM_MODEL` | `llama3.1:8b` | Model name |
| `MA_ALLOWED_LINK_DOMAINS` | empty | Domains answers may link to, e.g. `vendor.com` |
| `MA_ALLOWED_ORIGINS` | localhost | CORS allowlist |
| `MA_RATE_LIMIT_PER_MINUTE` | 20 | Per client IP |

## Evaluation

```bash
python eval/run_eval.py          # retrieval check, no model needed
python eval/run_eval.py --llm    # full answers through the configured model
```

`eval/questions.yaml` holds control questions with the expected source and key terms, including questions the manuals do not answer. On the sample manuals with the offline embedder: **13/14** pass. The failure is a known limit of keyword matching: an off-topic question about a ship's hull still matches the converter manual weakly. Extend the list to 30+ questions for your own documents before relying on the assistant.

## Tests

```bash
pytest
```

28 tests, no network or model needed.

## Data

`samples/manuals/` contains two **fictional** manuals (a frequency converter and a seawater pump) written for testing. `tests/fixtures/` contains the two planted attack documents. Do not commit real manuals or customer documents: `data/` is git-ignored.

## Limits

- Pattern-based scanning is easy to evade by paraphrasing. That is why it is only the first layer.
- The output checks catch links, contacts and leaks. They do not catch a wrong torque value that a poisoned document states as fact. Restrict who can add documents, and keep the quarantine review.
- Scanned PDFs without a text layer need OCR first.
