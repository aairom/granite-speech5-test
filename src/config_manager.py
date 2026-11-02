"""
Configuration management system.

Loads and merges configuration from three layers (lowest → highest priority):
  1. config/config.yaml  – project defaults
  2. .env file           – secrets (API keys, etc.)
  3. Environment variables – runtime overrides

The final merged config is exposed as a ConfigManager singleton so any module
can access it with ``from src.config_manager import get_config``.

Usage
-----
>>> from src.config_manager import get_config
>>> cfg = get_config()
>>> cfg.llm.backend
'ollama'
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Config data classes ────────────────────────────────────────────────────────

@dataclass
class SpeechPerfConfig:
    """Maps to config.yaml [speech.perf]."""
    max_new_tokens_short: int = 64
    short_clip_threshold_s: float = 10.0
    num_beams: int = 1
    use_cache: bool = True
    int8_quantize: bool = False
    compile: bool = False
    profile: bool = True


@dataclass
class SpeechConfig:
    model_name: str = "ibm-granite/granite-speech-5.0-2b"
    device: str = "auto"
    dtype: str = "auto"
    cache_dir: str = ""
    max_new_tokens: int = 512
    perf: SpeechPerfConfig = field(default_factory=SpeechPerfConfig)


@dataclass
class LLMBackendEntry:
    """Configuration for a single LLM backend."""
    base_url: str = ""
    model: str = ""


@dataclass
class LLMConfig:
    backend: str = "ollama"
    ollama: LLMBackendEntry = field(
        default_factory=lambda: LLMBackendEntry(
            base_url="http://localhost:11434/v1", model="granite4.2:latest"
        )
    )
    lmstudio: LLMBackendEntry = field(
        default_factory=lambda: LLMBackendEntry(
            base_url="http://localhost:1234/v1", model="local-model"
        )
    )
    llamacpp: LLMBackendEntry = field(
        default_factory=lambda: LLMBackendEntry(
            base_url="http://localhost:9931/v1", model="granite"
        )
    )
    openai: LLMBackendEntry = field(
        default_factory=lambda: LLMBackendEntry(
            base_url="https://api.openai.com/v1", model="gpt-4o-mini"
        )
    )
    anthropic: LLMBackendEntry = field(
        default_factory=lambda: LLMBackendEntry(
            base_url="https://api.anthropic.com", model="claude-3-haiku-20240307"
        )
    )
    temperature: float = 0.3
    max_tokens: int = 1024
    system_prompt: str = (
        "You are an AI assistant that processes speech transcriptions. "
        "When given a transcript, you may be asked to summarise, answer questions, "
        "translate, or perform other language tasks. "
        "Always be concise, accurate, and helpful."
    )

    def active_backend_entry(self) -> LLMBackendEntry:
        """Return the entry for the currently selected backend."""
        return getattr(self, self.backend.lower(), LLMBackendEntry())


@dataclass
class AppConfig:
    server_port: int = 7860
    share: bool = False
    log_level: str = "INFO"
    output_dir: str = "output"
    # TLS/HTTPS — enables browser microphone access on localhost
    ssl_certfile: str = ""   # e.g. "certs/cert.pem"
    ssl_keyfile: str = ""    # e.g. "certs/key.pem"


@dataclass
class Config:
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    app: AppConfig = field(default_factory=AppConfig)


# ── Bool coercion helper ──────────────────────────────────────────────────────

def _as_bool(val: Any, default: bool = False) -> bool:
    """
    Convert a value to bool safely.

    Handles:
      - Real booleans:  True/False → as-is
      - Strings:        "true"/"yes"/"1" → True;  "false"/"no"/"0" → False
      - None:           returns ``default``
      - Integers:       0 → False, non-zero → True

    This prevents the classic ``bool("false") == True`` pitfall that occurs
    when the minimal YAML parser fallback isn't available and raw string values
    slip into the config dict.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


# ── YAML loader (stdlib-only fallback) ────────────────────────────────────────

def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file using PyYAML if available, else fall back to a simple
    key:value parser that handles the project's config format.
    """
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        logger.warning(
            "PyYAML not installed; falling back to minimal YAML parser. "
            "Install it with: uv pip install pyyaml"
        )
        return _minimal_yaml_parser(path)


def _minimal_yaml_parser(path: Path) -> dict[str, Any]:
    """
    Extremely minimal YAML parser that handles the project's config.yaml.
    Supports:  scalars, nested sections (indented keys), quoted strings,
               booleans (true/false), integers, empty values.
    """
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(0, result)]
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip()
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(stripped)
            if ":" not in stripped:
                continue
            key, _, rest = stripped.partition(":")
            value_str = rest.strip().split("#")[0].strip()  # strip inline comment

            # Pop stack to the right indent level
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]

            # Parse value
            if value_str in ("", "null"):
                value: Any = None
            elif value_str.startswith('"') and value_str.endswith('"'):
                value = value_str[1:-1]
            elif value_str.startswith("'") and value_str.endswith("'"):
                value = value_str[1:-1]
            elif value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str  # plain string

            parent[key.strip()] = value
            if value is None:
                # This key will be a section; push it
                parent[key.strip()] = {}
                stack.append((indent + 2, parent[key.strip()]))

    return result


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file and return a dict of key→value strings."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                env[key.strip()] = val
    return env


# ── ConfigManager ─────────────────────────────────────────────────────────────

class ConfigManager:
    """
    Singleton-style configuration manager.

    Priority (highest wins): environment variables > .env file > config.yaml.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        env_path: Optional[Path] = None,
    ) -> None:
        root = Path(__file__).parent.parent
        self._config_path = config_path or (root / "config" / "config.yaml")
        self._env_path = env_path or (root / ".env")
        self._config: Optional[Config] = None

    def load(self) -> "ConfigManager":
        """Load all config sources and build the Config object."""
        raw = _load_yaml(self._config_path)
        dotenv = _load_dotenv(self._env_path)

        # Inject .env values into os.environ so they propagate uniformly
        for k, v in dotenv.items():
            os.environ.setdefault(k, v)

        self._config = self._build_config(raw)
        self._apply_env_overrides(self._config)
        self._configure_logging(self._config.app.log_level)
        return self

    def get(self) -> Config:
        """Return the parsed Config (auto-loads if not yet loaded)."""
        if self._config is None:
            self.load()
        return self._config  # type: ignore[return-value]

    # ── build from raw dict ────────────────────────────────────────────────────
    @staticmethod
    def _build_config(raw: dict[str, Any]) -> Config:
        cfg = Config()

        # Speech section
        s = raw.get("speech", {})
        if isinstance(s, dict):
            cfg.speech.model_name = s.get("model_name", cfg.speech.model_name)
            cfg.speech.device = s.get("device", cfg.speech.device)
            cfg.speech.dtype = s.get("dtype", cfg.speech.dtype)
            cfg.speech.cache_dir = s.get("cache_dir", cfg.speech.cache_dir) or ""
            cfg.speech.max_new_tokens = int(s.get("max_new_tokens", cfg.speech.max_new_tokens))
            # Speech perf sub-section
            p = s.get("perf", {})
            if isinstance(p, dict):
                d = cfg.speech.perf
                d.max_new_tokens_short = int(p.get("max_new_tokens_short", d.max_new_tokens_short))
                d.short_clip_threshold_s = float(p.get("short_clip_threshold_s", d.short_clip_threshold_s))
                d.num_beams = int(p.get("num_beams", d.num_beams))
                # Use _as_bool to guard against "false" string → True pitfall
                d.use_cache = _as_bool(p.get("use_cache"), default=d.use_cache)
                d.int8_quantize = _as_bool(p.get("int8_quantize"), default=d.int8_quantize)
                d.compile = _as_bool(p.get("compile"), default=d.compile)
                d.profile = _as_bool(p.get("profile"), default=d.profile)

        # LLM section
        ll = raw.get("llm", {})
        if isinstance(ll, dict):
            cfg.llm.backend = ll.get("backend", cfg.llm.backend)
            cfg.llm.temperature = float(ll.get("temperature", cfg.llm.temperature))
            cfg.llm.max_tokens = int(ll.get("max_tokens", cfg.llm.max_tokens))
            cfg.llm.system_prompt = ll.get("system_prompt", cfg.llm.system_prompt) or cfg.llm.system_prompt
            for name in ("ollama", "lmstudio", "llamacpp", "openai", "anthropic"):
                sub = ll.get(name, {})
                if isinstance(sub, dict):
                    entry = LLMBackendEntry(
                        base_url=sub.get("base_url", getattr(cfg.llm, name).base_url),
                        model=sub.get("model", getattr(cfg.llm, name).model),
                    )
                    setattr(cfg.llm, name, entry)

        # App section
        ap = raw.get("app", {})
        if isinstance(ap, dict):
            cfg.app.server_port = int(ap.get("server_port", cfg.app.server_port))
            cfg.app.share = _as_bool(ap.get("share"), default=cfg.app.share)
            cfg.app.log_level = ap.get("log_level", cfg.app.log_level)
            cfg.app.output_dir = ap.get("output_dir", cfg.app.output_dir) or "output"
            cfg.app.ssl_certfile = ap.get("ssl_certfile", cfg.app.ssl_certfile) or ""
            cfg.app.ssl_keyfile = ap.get("ssl_keyfile", cfg.app.ssl_keyfile) or ""

        return cfg

    # ── env overrides ─────────────────────────────────────────────────────────
    @staticmethod
    def _apply_env_overrides(cfg: Config) -> None:
        """Map well-known environment variables into the config object."""
        overrides = {
            "LLM_BACKEND": ("llm", "backend"),
            "SPEECH_DEVICE": ("speech", "device"),
            "SPEECH_DTYPE": ("speech", "dtype"),
            "SPEECH_MODEL": ("speech", "model_name"),
            "SPEECH_MAX_NEW_TOKENS": ("speech", "max_new_tokens"),
            # perf sub-keys
            "SPEECH_INT8": None,        # handled below
            "SPEECH_COMPILE": None,
            "SPEECH_NUM_BEAMS": None,
            "SPEECH_PROFILE": None,
            "APP_PORT": ("app", "server_port"),
            "APP_LOG_LEVEL": ("app", "log_level"),
            "APP_SSL_CERTFILE": ("app", "ssl_certfile"),
            "APP_SSL_KEYFILE": ("app", "ssl_keyfile"),
            "LLM_OLLAMA_MODEL": None,   # handled below
            "LLM_LMSTUDIO_MODEL": None,
            "LLM_LLAMACPP_MODEL": None,
            "LLM_OPENAI_MODEL": None,
            "LLM_ANTHROPIC_MODEL": None,
            "LLM_OLLAMA_BASE_URL": None,
            "LLM_LMSTUDIO_BASE_URL": None,
            "LLM_LLAMACPP_BASE_URL": None,
        }
        for env_key, target in overrides.items():
            val = os.environ.get(env_key)
            if val is None:
                continue
            if target:
                section_name, attr = target
                section = getattr(cfg, section_name)
                # Convert type if needed
                old = getattr(section, attr, None)
                if isinstance(old, int):
                    val = int(val)  # type: ignore[assignment]
                elif isinstance(old, bool):
                    val = val.lower() in ("1", "true", "yes")  # type: ignore[assignment]
                setattr(section, attr, val)

        # Speech perf env overrides
        perf = cfg.speech.perf
        if os.environ.get("SPEECH_INT8"):
            perf.int8_quantize = os.environ["SPEECH_INT8"].lower() in ("1", "true", "yes")
        if os.environ.get("SPEECH_COMPILE"):
            perf.compile = os.environ["SPEECH_COMPILE"].lower() in ("1", "true", "yes")
        if os.environ.get("SPEECH_NUM_BEAMS"):
            perf.num_beams = int(os.environ["SPEECH_NUM_BEAMS"])
        if os.environ.get("SPEECH_PROFILE"):
            perf.profile = os.environ["SPEECH_PROFILE"].lower() in ("1", "true", "yes")

        # Backend-specific model/url overrides
        for backend in ("ollama", "lmstudio", "llamacpp", "openai", "anthropic"):
            model_env = f"LLM_{backend.upper()}_MODEL"
            url_env = f"LLM_{backend.upper()}_BASE_URL"
            entry: LLMBackendEntry = getattr(cfg.llm, backend)
            if os.environ.get(model_env):
                entry.model = os.environ[model_env]
            if os.environ.get(url_env):
                entry.base_url = os.environ[url_env]

    @staticmethod
    def _configure_logging(level: str) -> None:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[ConfigManager] = None


def get_config() -> Config:
    """
    Return the global Config singleton (loads on first call).

    Usage
    -----
    >>> from src.config_manager import get_config
    >>> cfg = get_config()
    >>> cfg.llm.backend
    'ollama'
    """
    global _manager
    if _manager is None:
        _manager = ConfigManager()
    return _manager.get()


def reload_config() -> Config:
    """Force a reload of the configuration from disk."""
    global _manager
    _manager = ConfigManager()
    return _manager.get()
