"""
Unit tests for the LLM client abstraction layer.
Uses only stdlib urllib mocking – no real network calls.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from io import BytesIO
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import (
    OllamaClient,
    LMStudioClient,
    LlamaCppClient,
    OpenAIClient,
    AnthropicClient,
    LLMRequest,
    LLMResponse,
    build_llm_client,
    list_backends,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_openai_response(text: str = "Hello!", model: str = "test-model") -> bytes:
    return json.dumps({
        "choices": [{"message": {"content": text}}],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }).encode()


def _fake_anthropic_response(text: str = "Hello!") -> bytes:
    return json.dumps({
        "content": [{"type": "text", "text": text}],
        "model": "claude-3-haiku",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()


@contextmanager
def _mock_urlopen(response_bytes: bytes):
    """Context manager that patches urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_bytes
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("src.llm_client.urllib.request.urlopen", return_value=mock_resp):
        yield mock_resp


# ── Tests for OllamaClient ─────────────────────────────────────────────────────

class TestOllamaClient(unittest.TestCase):
    def test_default_init(self):
        client = OllamaClient()
        self.assertEqual(client.backend_name, "ollama")
        self.assertIn("11434", client.base_url)

    def test_chat_returns_response(self):
        client = OllamaClient(model="granite3.3:8b")
        with _mock_urlopen(_fake_openai_response("Test reply", "granite3.3:8b")):
            resp = client.chat([{"role": "user", "content": "Hello"}])
        self.assertIsInstance(resp, LLMResponse)
        self.assertEqual(resp.text, "Test reply")
        self.assertEqual(resp.backend, "ollama")
        self.assertGreater(resp.latency_s, 0)

    def test_chat_with_system_prompt(self):
        client = OllamaClient(model="granite3.3:8b")
        with _mock_urlopen(_fake_openai_response("With system")):
            resp = client.chat(
                [{"role": "user", "content": "Hi"}],
                system_prompt="You are helpful.",
            )
        self.assertEqual(resp.text, "With system")

    def test_token_counts(self):
        client = OllamaClient()
        with _mock_urlopen(_fake_openai_response()):
            resp = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.input_tokens, 10)
        self.assertEqual(resp.output_tokens, 5)


# ── Tests for LMStudioClient ───────────────────────────────────────────────────

class TestLMStudioClient(unittest.TestCase):
    def test_default_init(self):
        client = LMStudioClient()
        self.assertEqual(client.backend_name, "lmstudio")
        self.assertIn("1234", client.base_url)

    def test_chat_success(self):
        client = LMStudioClient(model="mistral-7b")
        with _mock_urlopen(_fake_openai_response("LM Studio reply", "mistral-7b")):
            resp = client.chat([{"role": "user", "content": "What is 2+2?"}])
        self.assertEqual(resp.text, "LM Studio reply")


# ── Tests for LlamaCppClient ──────────────────────────────────────────────────

class TestLlamaCppClient(unittest.TestCase):
    def test_default_port(self):
        client = LlamaCppClient()
        self.assertIn("9931", client.base_url)
        self.assertEqual(client.backend_name, "llamacpp")


# ── Tests for OpenAIClient ─────────────────────────────────────────────────────

class TestOpenAIClient(unittest.TestCase):
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Ensure OPENAI_API_KEY is absent
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            with self.assertRaises(ValueError):
                OpenAIClient()

    def test_init_with_explicit_key(self):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test-key")
        self.assertEqual(client.api_key, "sk-test-key")
        self.assertEqual(client.backend_name, "openai")

    def test_chat_success(self):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
        with _mock_urlopen(_fake_openai_response("OpenAI reply", "gpt-4o-mini")):
            resp = client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(resp.text, "OpenAI reply")

    def test_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}):
            client = OpenAIClient()
        self.assertEqual(client.api_key, "sk-env-key")


# ── Tests for AnthropicClient ─────────────────────────────────────────────────

class TestAnthropicClient(unittest.TestCase):
    def test_raises_without_api_key(self):
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                AnthropicClient()

    def test_chat_success(self):
        client = AnthropicClient(model="claude-3-haiku-20240307", api_key="sk-ant-test")
        with _mock_urlopen(_fake_anthropic_response("Anthropic reply")):
            resp = client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(resp.text, "Anthropic reply")
        self.assertEqual(resp.backend, "anthropic")

    def test_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env"}):
            client = AnthropicClient()
        self.assertEqual(client.api_key, "sk-ant-env")


# ── Tests for build_llm_client factory ───────────────────────────────────────

class TestBuildLLMClient(unittest.TestCase):
    def test_build_ollama(self):
        client = build_llm_client("ollama", model="granite3.3:8b")
        self.assertIsInstance(client, OllamaClient)

    def test_build_lmstudio(self):
        client = build_llm_client("lmstudio", model="local-model")
        self.assertIsInstance(client, LMStudioClient)

    def test_build_llamacpp(self):
        client = build_llm_client("llamacpp", model="granite")
        self.assertIsInstance(client, LlamaCppClient)

    def test_build_openai(self):
        client = build_llm_client("openai", model="gpt-4o-mini", api_key="sk-x")
        self.assertIsInstance(client, OpenAIClient)

    def test_build_anthropic(self):
        client = build_llm_client("anthropic", model="claude-3-haiku", api_key="sk-ant-x")
        self.assertIsInstance(client, AnthropicClient)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            build_llm_client("unknown_backend", model="x")

    def test_case_insensitive(self):
        client = build_llm_client("OLLAMA", model="granite3.3:8b")
        self.assertIsInstance(client, OllamaClient)

    def test_base_url_override(self):
        client = build_llm_client(
            "ollama", model="granite3.3:8b", base_url="http://custom:9999/v1"
        )
        self.assertIn("9999", client.base_url)

    def test_list_backends_returns_all(self):
        backends = list_backends()
        for b in ("ollama", "lmstudio", "llamacpp", "openai", "anthropic"):
            self.assertIn(b, backends)


# ── Tests for health_check ────────────────────────────────────────────────────

class TestHealthCheck(unittest.TestCase):
    def test_health_check_true_on_success(self):
        client = OllamaClient()
        with _mock_urlopen(_fake_openai_response("pong")):
            result = client.health_check()
        self.assertTrue(result)

    def test_health_check_false_on_failure(self):
        client = OllamaClient()
        with patch("src.llm_client.urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = client.health_check()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
