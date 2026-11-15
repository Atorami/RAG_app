# Threat model

Scope: a technician asks questions; the assistant answers from a folder of manuals using a local or remote LLM. Mapped to the [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/).

## Assets

- Correct, safe maintenance instructions (a wrong step can injure someone or damage equipment).
- Confidential documents: customer drawings, internal procedures, vessel data.
- The technician's trust: links and contacts in answers are followed without much thought.

## Trust boundaries

| Input | Trusted? | Why |
| --- | --- | --- |
| Technician's question | Mostly | Authenticated user, but may paste text from elsewhere |
| Manuals in the docs folder | **No** | Supplier PDFs, scans, e-mailed bulletins: anyone upstream can plant text |
| Model output | **No** | Can be steered by the documents it reads |

## Threats and controls

| OWASP | Threat in this system | Control | Test |
| --- | --- | --- | --- |
| LLM01 Prompt injection (indirect) | A planted bulletin tells the model to send technicians to an attacker's e-mail or to disable the parameter lock | Ingest scan and quarantine; sources wrapped as data with escaped tags; output checks | `test_layer1_*`, `test_layer2_*`, `test_source_cannot_close_its_own_tag` |
| LLM01 Hidden text | Zero-width or bidi characters hide instructions from a human reviewer | Hidden characters detected and stripped before indexing | `test_detects_hidden_characters_*` |
| LLM02 Sensitive information disclosure | Customer documents sent to a cloud LLM | Local model by default; `data/` git-ignored; no documents in the repo | — |
| LLM05 Improper output handling | Answer contains HTML/JS rendered in the browser | UI uses `textContent` only | manual review |
| LLM06 Excessive agency | Model takes actions | The assistant has no tools: it can only return text | by design |
| LLM07 System prompt leakage | Injected text makes the model print its rules | Random canary in the system prompt; answers containing it are blocked | `test_blocks_system_prompt_leak` |
| LLM08 Vector and embedding weaknesses | Poisoned chunk ranks high for common questions | Quarantined chunks never returned by search; index stored without pickle | `test_poisoned_chunk_is_quarantined_*`, `test_index_round_trip_without_pickle` |
| LLM09 Misinformation | Model answers beyond the documents | Score threshold → refusal; answers must cite; citations must exist | `test_refuses_*`, `test_blocks_citation_*` |
| LLM10 Unbounded consumption | Huge questions or request floods | 500-character limit, per-IP rate limit, `max_tokens` cap | `test_rejects_overlong_question`, `test_rate_limit_kicks_in` |

## Residual risks

1. **Paraphrased injection.** The ingest scanner matches patterns; a reworded instruction passes. The output checks catch its usual goals (links, contacts, leaks), but not every goal.
2. **Poisoned facts.** A document that simply states a wrong value ("torque 25 Nm" instead of 2.5) is not an injection and is not detected. Control: restrict who can add documents and keep a review step.
3. **Model-specific behaviour.** Defences were tested with a fake model that always obeys injections. Run `eval/run_eval.py --llm` and red-team tests against the actual model you deploy.

## Next steps

- Automated red-teaming of the deployed model with garak or PyRIT.
- Provenance per document (who added it, checksum) and a signed allowlist of approved sources.
- Authentication on the API before exposing it beyond localhost.
