"""Chat model clients."""

from __future__ import annotations

from typing import Protocol

import httpx


class LLMUnavailable(RuntimeError):
    """The chat model could not be reached or returned an error."""


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatibleChat:
    """Works with Ollama, LM Studio, vLLM, OpenAI and other /v1/chat/completions servers."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 60, max_tokens: int = 500) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        try:
            return self._complete(system, user)
        except httpx.HTTPError as err:
            raise LLMUnavailable(f"Chat model at {self.url} is not reachable: {err}") from err

    def _complete(self, system: str, user: str) -> str:
        response = httpx.post(
            self.url,
            headers=self.headers,
            timeout=self.timeout_s,
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
