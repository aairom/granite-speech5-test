"""
LLM backend abstraction layer.

Provides a unified LLMClient interface over multiple local and cloud backends:

  Local
  -----
  • Ollama   – http://localhost:11434 (default)
  • LM Studio – http://localhost:1234  (OpenAI-compatible endpoint)
  • llama.cpp server – http://localhost:9931  (project rule: llama.cpp port)

  Cloud
  -----
  • OpenAI   – https://api.openai.com/v1
  • Anthropic – Claude via Messages API
  • IBM watsonx.ai – https://us-south.ml.cloud.ibm.com (optional)

All backends expose the same .chat(messages) → str interface so the
application layer never needs to know which backend is active.

Usage
-----
>>> from src.llm_client import build_llm_client
>>> llm = build_llm_client(backend="ollama", model="granite3.3:8b")
>>> llm.chat([{"role": "user", "content": "Summarise this transcript: …"}])
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request  # imported at module level for testability
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Request / Response containers ─────────────────────────────────────────────

@dataclass
class LLMRequest:
    messages: list[dict[str, str]]
    model: str
    temperature: float = 0.3
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


# ── Abstract base ──────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Abstract base class for all LLM backend clients."""

    backend_name: str = "base"

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self._extra = kwargs

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Send a chat completion request and return a structured response.

        Parameters
        ----------
        messages : list of {"role": "user"|"assistant"|"system", "content": str}
        system_prompt : optional system message prepended to messages
        temperature : sampling temperature (0 = deterministic)
        max_tokens : maximum output tokens
        """
        req = LLMRequest(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        t0 = time.perf_counter()
        resp = self._send(req)
        resp.latency_s = time.perf_counter() - t0
        logger.debug(
            "[%s] %.2f s  in=%d out=%d",
            self.backend_name,
            resp.latency_s,
            resp.input_tokens,
            resp.output_tokens,
        )
        return resp

    @abstractmethod
    def _send(self, req: LLMRequest) -> LLMResponse:
        """Backend-specific implementation."""
        ...

    def health_check(self) -> bool:
        """Return True when the backend is reachable; False otherwise."""
        try:
            self.chat([{"role": "user", "content": "ping"}], max_tokens=4)
            return True
        except Exception as exc:
            logger.warning("[%s] health check failed: %s", self.backend_name, exc)
            return False


# ── OpenAI-compatible backend (Ollama, LM Studio, llama.cpp, OpenAI) ──────────

class OpenAICompatibleClient(LLMClient):
    """
    Client for any server exposing an OpenAI-compatible /v1/chat/completions
    endpoint: Ollama, LM Studio, llama.cpp server, or OpenAI itself.

    Parameters
    ----------
    model : model identifier (e.g. "granite3.3:8b", "gpt-4o-mini")
    base_url : API base URL including /v1
    api_key : Bearer token (required for OpenAI; can be "ollama" for local)
    """

    backend_name = "openai-compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "ollama", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _send(self, req: LLMRequest) -> LLMResponse:
        messages = list(req.messages)
        if req.system_prompt:
            messages = [{"role": "system", "content": req.system_prompt}] + messages

        payload = json.dumps(
            {
                "model": req.model,
                "messages": messages,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            }
        ).encode()

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=120) as resp:
            body = json.loads(resp.read())

        choice = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return LLMResponse(
            text=choice.strip(),
            model=body.get("model", req.model),
            backend=self.backend_name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=body,
        )


class OllamaClient(OpenAICompatibleClient):
    """Ollama local server (http://localhost:11434)."""

    backend_name = "ollama"

    def __init__(self, model: str = "granite3.3:8b", base_url: str = "http://localhost:11434/v1", **kwargs: Any) -> None:
        super().__init__(model=model, base_url=base_url, api_key="ollama", **kwargs)


class LMStudioClient(OpenAICompatibleClient):
    """LM Studio local server (http://localhost:1234)."""

    backend_name = "lmstudio"

    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1", **kwargs: Any) -> None:
        super().__init__(model=model, base_url=base_url, api_key="lm-studio", **kwargs)


class LlamaCppClient(OpenAICompatibleClient):
    """llama.cpp HTTP server (http://localhost:9931 per project rules)."""

    backend_name = "llamacpp"

    def __init__(self, model: str = "granite", base_url: str = "http://localhost:9931/v1", **kwargs: Any) -> None:
        super().__init__(model=model, base_url=base_url, api_key="llamacpp", **kwargs)


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI cloud API (requires OPENAI_API_KEY env variable or explicit api_key)."""

    backend_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        import os
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your environment "
                "or pass api_key= to OpenAIClient()."
            )
        super().__init__(model=model, base_url=base_url, api_key=resolved_key, **kwargs)


# ── Anthropic backend ─────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """
    Anthropic Claude via the Messages API.
    Requires ANTHROPIC_API_KEY env variable or explicit api_key parameter.
    """

    backend_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in your "
                "environment or pass api_key= to AnthropicClient()."
            )

    def _send(self, req: LLMRequest) -> LLMResponse:
        messages = list(req.messages)
        system = req.system_prompt or ""

        payload = json.dumps(
            {
                "model": req.model,
                "max_tokens": req.max_tokens,
                "system": system,
                "messages": [
                    m for m in messages if m["role"] != "system"
                ],
                "temperature": req.temperature,
            }
        ).encode()

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=120) as resp:
            body = json.loads(resp.read())

        text = body["content"][0]["text"].strip()
        usage = body.get("usage", {})
        return LLMResponse(
            text=text,
            model=body.get("model", req.model),
            backend=self.backend_name,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw=body,
        )


# ── Factory function ───────────────────────────────────────────────────────────

_BACKEND_MAP: dict[str, type[LLMClient]] = {
    "ollama": OllamaClient,
    "lmstudio": LMStudioClient,
    "llamacpp": LlamaCppClient,
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def build_llm_client(
    backend: str,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """
    Build and return an LLMClient for the requested backend.

    Parameters
    ----------
    backend : one of "ollama" | "lmstudio" | "llamacpp" | "openai" | "anthropic"
    model   : model identifier string passed to the backend
    base_url : override the default base URL (for custom deployments)
    api_key  : override the default API key / token
    **kwargs : extra keyword arguments forwarded to the concrete client class

    Example
    -------
    >>> client = build_llm_client("ollama", model="granite3.3:8b")
    >>> response = client.chat([{"role": "user", "content": "Hello!"}])
    >>> print(response.text)
    """
    backend_lower = backend.lower()
    if backend_lower not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Supported: {list(_BACKEND_MAP.keys())}"
        )

    cls = _BACKEND_MAP[backend_lower]
    init_kwargs: dict[str, Any] = {"model": model, **kwargs}
    if base_url:
        init_kwargs["base_url"] = base_url
    if api_key:
        init_kwargs["api_key"] = api_key

    client = cls(**init_kwargs)
    logger.info("LLM client: backend=%s  model=%s", backend_lower, model)
    return client


def list_backends() -> list[str]:
    """Return list of registered backend names."""
    return list(_BACKEND_MAP.keys())
