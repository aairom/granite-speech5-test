"""
Unit tests for the configuration management system.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import (
    Config,
    ConfigManager,
    LLMBackendEntry,
    LLMConfig,
    SpeechConfig,
    AppConfig,
    _as_bool,
    _load_dotenv,
    _minimal_yaml_parser,
    get_config,
    reload_config,
)


# ── Helper: write temp YAML files ─────────────────────────────────────────────

def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content)


def _write_dotenv(path: Path, content: str) -> None:
    path.write_text(content)


# ── Tests for _minimal_yaml_parser ───────────────────────────────────────────

class TestMinimalYamlParser(unittest.TestCase):
    def test_parses_scalars(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
            fh.write('speech:\n  model_name: "custom-model"\n  device: auto\n')
            fpath = Path(fh.name)
        try:
            result = _minimal_yaml_parser(fpath)
            self.assertIn("speech", result)
        finally:
            fpath.unlink()

    def test_ignores_comments(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
            fh.write("# comment\nkey: value\n")
            fpath = Path(fh.name)
        try:
            result = _minimal_yaml_parser(fpath)
            self.assertIn("key", result)
            self.assertEqual(result["key"], "value")
        finally:
            fpath.unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
            fpath = Path(fh.name)
        try:
            result = _minimal_yaml_parser(fpath)
            self.assertEqual(result, {})
        finally:
            fpath.unlink()


# ── Tests for _load_dotenv ─────────────────────────────────────────────────────

class TestLoadDotenv(unittest.TestCase):
    def test_basic_parsing(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as fh:
            fh.write('OPENAI_API_KEY=sk-test\nANTHROPIC_API_KEY="sk-ant-test"\n')
            fpath = Path(fh.name)
        try:
            env = _load_dotenv(fpath)
            self.assertEqual(env["OPENAI_API_KEY"], "sk-test")
            self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-ant-test")
        finally:
            fpath.unlink()

    def test_ignores_comments(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False
        ) as fh:
            fh.write("# comment\nKEY=VALUE\n")
            fpath = Path(fh.name)
        try:
            env = _load_dotenv(fpath)
            self.assertNotIn("# comment", env)
            self.assertEqual(env["KEY"], "VALUE")
        finally:
            fpath.unlink()

    def test_missing_file_returns_empty(self):
        env = _load_dotenv(Path("/nonexistent/.env"))
        self.assertEqual(env, {})


# ── Tests for ConfigManager ────────────────────────────────────────────────────

class TestConfigManager(unittest.TestCase):
    def _make_manager(self, yaml_content: str = "", env_content: str = "") -> ConfigManager:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cfg_path = tmpdir / "config.yaml"
            env_path = tmpdir / ".env"
            cfg_path.write_text(yaml_content)
            env_path.write_text(env_content)
            manager = ConfigManager(config_path=cfg_path, env_path=env_path)
            manager.load()
            return manager

    def test_defaults_returned_with_empty_yaml(self):
        manager = self._make_manager("")
        cfg = manager.get()
        self.assertIsInstance(cfg, Config)
        self.assertEqual(cfg.speech.model_name, "ibm-granite/granite-speech-5.0-2b")
        self.assertEqual(cfg.llm.backend, "ollama")

    def test_yaml_overrides_defaults(self):
        yaml = (
            "speech:\n"
            '  model_name: "custom-model"\n'
            '  device: "cuda"\n'
            "llm:\n"
            '  backend: "openai"\n'
        )
        manager = self._make_manager(yaml)
        cfg = manager.get()
        self.assertEqual(cfg.speech.model_name, "custom-model")
        self.assertEqual(cfg.speech.device, "cuda")
        self.assertEqual(cfg.llm.backend, "openai")

    def test_env_overrides_yaml(self):
        yaml = "llm:\n  backend: \"ollama\"\n"
        with patch.dict(os.environ, {"LLM_BACKEND": "anthropic"}):
            manager = self._make_manager(yaml)
        cfg = manager.get()
        self.assertEqual(cfg.llm.backend, "anthropic")

    def test_active_backend_entry(self):
        yaml = (
            "llm:\n"
            '  backend: "ollama"\n'
            "  ollama:\n"
            '    model: "granite3.3:8b"\n'
            '    base_url: "http://localhost:11434/v1"\n'
        )
        manager = self._make_manager(yaml)
        cfg = manager.get()
        entry = cfg.llm.active_backend_entry()
        self.assertEqual(entry.model, "granite3.3:8b")

    def test_app_port_parsed(self):
        yaml = "app:\n  server_port: 7890\n"
        manager = self._make_manager(yaml)
        cfg = manager.get()
        self.assertEqual(cfg.app.server_port, 7890)

    def test_max_new_tokens_int(self):
        yaml = "speech:\n  max_new_tokens: 256\n"
        manager = self._make_manager(yaml)
        cfg = manager.get()
        self.assertIsInstance(cfg.speech.max_new_tokens, int)
        self.assertEqual(cfg.speech.max_new_tokens, 256)


# ── Tests for _as_bool helper ─────────────────────────────────────────────────

class TestAsBool(unittest.TestCase):
    """Verify _as_bool is immune to the bool("false") == True pitfall."""

    def test_true_string(self):
        for v in ("true", "True", "TRUE", "1", "yes", "on"):
            with self.subTest(v=v):
                self.assertTrue(_as_bool(v))

    def test_false_string(self):
        for v in ("false", "False", "FALSE", "0", "no", "off"):
            with self.subTest(v=v):
                self.assertFalse(_as_bool(v))

    def test_native_bool_passthrough(self):
        self.assertTrue(_as_bool(True))
        self.assertFalse(_as_bool(False))

    def test_none_uses_default_false(self):
        self.assertFalse(_as_bool(None, default=False))

    def test_none_uses_default_true(self):
        self.assertTrue(_as_bool(None, default=True))

    def test_integer(self):
        self.assertTrue(_as_bool(1))
        self.assertFalse(_as_bool(0))

    def test_bool_false_string_was_pitfall(self):
        """Explicit regression: bool("false") == True but _as_bool("false") == False."""
        # This is the bug this function was introduced to prevent
        self.assertTrue(bool("false"))    # Python built-in is broken for this
        self.assertFalse(_as_bool("false"))  # our helper is correct

    def test_perf_false_via_string(self):
        """End-to-end: profile:false as a string must not enable profiling."""
        raw = {"speech": {"perf": {"profile": "false", "int8_quantize": "false"}}}
        mgr = ConfigManager.__new__(ConfigManager)
        cfg = ConfigManager._build_config(raw)
        self.assertFalse(cfg.speech.perf.profile)
        self.assertFalse(cfg.speech.perf.int8_quantize)


# ── Tests for module-level get_config / reload_config ─────────────────────────

class TestModuleLevelAPI(unittest.TestCase):
    def test_get_config_returns_config(self):
        # Patch the manager so we don't read real files
        with patch("src.config_manager._manager", None):
            cfg = get_config()
        self.assertIsInstance(cfg, Config)

    def test_reload_config_resets_manager(self):
        import src.config_manager as cm
        original_manager = cm._manager
        with patch("src.config_manager._manager", None):
            cfg = reload_config()
        self.assertIsInstance(cfg, Config)


if __name__ == "__main__":
    unittest.main()
