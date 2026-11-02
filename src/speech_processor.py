"""
IBM Granite Speech 5.0-2B — Core speech processing module.

Model reference: https://huggingface.co/ibm-granite/granite-speech-5.0-2b
Architecture: two-pass design (CTC speech encoder → Granite-4.0-1B LLM backbone)
Supported tasks: ASR (transcription) and AST (translation) in 6 languages.

Usage constraints (Apache 2.0 licence, no additional restrictions):
- Audio must be 16 kHz mono; longer recordings should be split into ≤2-minute segments.
- Prompt tokens use the <|audio|> placeholder which the processor replaces with audio embeddings.

Performance guide
-----------------
On CPU (M-series or x86) a 2-second clip takes ~60-120 s with float32 because the
Granite-4.0-1B LLM backbone must run greedy-decode token-by-token.  To reach
real-time speed, one of the following is needed:

  1. CUDA GPU (any card ≥ 4 GB VRAM) → bfloat16 → 1-5× real-time
  2. Apple Silicon with MPS           → float16  → ~5× real-time
  3. INT-8 dynamic quantisation       → ~2× CPU speedup (built-in, no GPU)
  4. Reduce max_new_tokens            → linear speedup (safe for short clips)
  5. ONNX runtime (WebGPU / CoreML)   → best CPU/edge throughput

All optimisation flags are controlled via config/config.yaml [speech.perf].
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── torchaudio: optional, version-guarded ─────────────────────────────────────
# torchaudio≥2.9 (released Oct 2025) removed its bundled audio backends
# (sox / soundfile / ffmpeg) and made torchaudio.load() a thin alias to
# load_with_torchcodec(), which requires torchcodec + FFmpeg.  We therefore
# treat torchaudio as a soft dependency and fall through to soundfile/scipy
# when it is absent or too new.  See _load_audio_file() below.
try:
    import torchaudio as _torchaudio
    _torchaudio_version = tuple(
        int(x) for x in _torchaudio.__version__.split(".")[:2]
        if x.isdigit()
    )
    # ≥2.9 routes .load() through torchcodec — disable for audio I/O
    _TORCHAUDIO_IO_OK = _torchaudio_version < (2, 9)
    _TORCHAUDIO_RESAMPLE_OK = True   # Resample transform is fine in all versions
except ImportError:
    _torchaudio = None          # type: ignore[assignment]
    _torchaudio_version = (0, 0)
    _TORCHAUDIO_IO_OK = False
    _TORCHAUDIO_RESAMPLE_OK = False

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "ibm-granite/granite-speech-5.0-2b"
TARGET_SAMPLE_RATE = 16_000           # Hz required by the model
MAX_SEGMENT_SECONDS = 119             # keep under the 2-minute recommended limit
SUPPORTED_LANGUAGES = ("en", "fr", "de", "es", "pt", "ja")

# Map BCP-47 codes → full language names expected by the model prompt.
# The official model card uses full names, e.g. "translate the speech to French."
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
    "it": "Italian",
    "zh": "Mandarin",
}

# Task-specific prompt templates — matched exactly to the official model card.
# Reference: https://huggingface.co/ibm-granite/granite-speech-5.0-2b
#
# ASR prompts:
#   "can you transcribe the speech into a written format?"
#   "transcribe the speech verbatim"
# AST prompts:
#   "translate the speech to <language>."   ← simple, canonical form
#
# IMPORTANT: {target_language} is substituted with the FULL language name
# (e.g. "French"), NOT the BCP-47 code (e.g. "fr").  Passing "fr" causes the
# model to ignore the instruction and output the source language instead.
PROMPT_TEMPLATES = {
    "transcribe": (
        "<|audio|>can you transcribe the speech into a written format?"
    ),
    "transcribe_verbatim": (
        "<|audio|>transcribe the speech verbatim."
    ),
    # Any language → English
    "translate_to_en": (
        "<|audio|>translate the speech to English."
    ),
    # English (or any supported source) → target language (full name)
    "translate_from_en": (
        "<|audio|>translate the speech to {target_language}."
    ),
}


# ── Audio I/O helpers (torchaudio-version-safe) ────────────────────────────────

def _load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file to a float32 numpy array + sample-rate.

    Backend priority
    ----------------
    1. ``torchaudio.load()`` — only if torchaudio<2.9 (older versions bundle
       sox/soundfile/ffmpeg and don't need torchcodec).
    2. ``soundfile.read()`` — pure-Python, no C++ codec required; works for
       WAV / FLAC / OGG / AIFF.
    3. ``scipy.io.wavfile.read()`` — stdlib-adjacent fallback for plain WAV.
    4. ``torchcodec``  — new TorchCodec AudioDecoder (torchaudio≥2.9 native).

    Raises ``RuntimeError`` when no backend can load the file.

    Why restart is needed after installation
    -----------------------------------------
    Python's import system caches module presence at import time.  If
    ``soundfile`` or ``torchcodec`` was missing when the interpreter started,
    it won't be visible to ``importlib.import_module`` calls in the same
    process — the cached ``sys.modules`` entry (or lack thereof) from startup
    is authoritative.  A **full process restart** (not a Gradio page refresh,
    which keeps the Python process alive) is required for newly installed
    packages to become importable.
    """
    path_str = str(path)

    # ── Backend 1: torchaudio <2.9 ────────────────────────────────────────────
    if _TORCHAUDIO_IO_OK and _torchaudio is not None:
        try:
            wav, sr = _torchaudio.load(path_str, normalize=True)
            return wav.numpy(), sr
        except Exception as exc:
            logger.debug("torchaudio.load failed (%s); trying next backend", exc)

    # ── Backend 2: soundfile ──────────────────────────────────────────────────
    try:
        import soundfile as sf  # type: ignore[import]
        data, sr = sf.read(path_str, dtype="float32", always_2d=True)
        # soundfile returns (samples, channels) → transpose to (channels, samples)
        return data.T, sr
    except ImportError:
        logger.debug("soundfile not installed; trying next backend")
    except Exception as exc:
        logger.debug("soundfile.read failed (%s); trying next backend", exc)

    # ── Backend 3: scipy.io.wavfile (WAV-only) ────────────────────────────────
    try:
        from scipy.io import wavfile  # type: ignore[import]
        sr, data = wavfile.read(path_str)
        # scipy returns int16/int32; normalise to float32 in [-1, 1]
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        data = data.astype(np.float32)
        if data.ndim == 1:
            data = data[np.newaxis, :]   # add channel dim
        else:
            data = data.T                # (channels, samples)
        return data, sr
    except ImportError:
        logger.debug("scipy not installed; trying next backend")
    except Exception as exc:
        logger.debug("scipy.io.wavfile failed (%s); trying next backend", exc)

    # ── Backend 4: torchcodec AudioDecoder (torchaudio ≥2.9) ─────────────────
    try:
        from torchcodec.decoders import AudioDecoder  # type: ignore[import]
        decoder = AudioDecoder(path_str)
        frames = decoder.get_all_samples()
        # frames.data shape: (channels, samples) as float32
        return frames.data.numpy(), frames.sample_rate
    except ImportError:
        logger.debug("torchcodec not installed; trying next backend")
    except Exception as exc:
        logger.debug("torchcodec AudioDecoder failed (%s)", exc)

    raise RuntimeError(
        f"Cannot load audio file '{path}'. "
        "No working audio backend found.\n\n"
        "Fix options (pick ONE):\n"
        "  A) Install soundfile (recommended, no FFmpeg needed):\n"
        "       uv pip install soundfile\n"
        "  B) Install torchcodec + FFmpeg (for torchaudio≥2.9):\n"
        "       uv pip install torchcodec\n"
        "       # and ensure FFmpeg 4-8 is in PATH\n"
        "  C) Pin torchaudio to a version that bundles its own backends:\n"
        "       uv pip install 'torchaudio<2.9'\n\n"
        "After installing, restart the application (a page refresh is NOT\n"
        "sufficient — Python caches imports at process start)."
    )


def _resample_numpy(wav_np: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Resample a (channels, samples) float32 array from src_sr → dst_sr.

    Uses ``torchaudio.transforms.Resample`` when available (GPU-accelerated),
    otherwise falls back to ``scipy.signal.resample_poly`` (CPU, exact).
    """
    if src_sr == dst_sr:
        return wav_np

    # torchaudio Resample works in all versions (transform, not I/O)
    if _TORCHAUDIO_RESAMPLE_OK and _torchaudio is not None:
        try:
            t = torch.from_numpy(wav_np)
            resampler = _torchaudio.transforms.Resample(src_sr, dst_sr)
            return resampler(t).numpy()
        except Exception as exc:
            logger.debug("torchaudio Resample failed (%s); falling back to scipy", exc)

    try:
        from scipy.signal import resample_poly  # type: ignore[import]
        import math
        g = math.gcd(src_sr, dst_sr)
        return resample_poly(wav_np, dst_sr // g, src_sr // g, axis=-1).astype(np.float32)
    except ImportError:
        pass

    # Last resort: linear interpolation via numpy
    ratio = dst_sr / src_sr
    n_out = int(wav_np.shape[-1] * ratio)
    channels = wav_np.shape[0] if wav_np.ndim == 2 else 1
    result = np.zeros((channels, n_out), dtype=np.float32)
    x_old = np.linspace(0, 1, wav_np.shape[-1])
    x_new = np.linspace(0, 1, n_out)
    src = wav_np if wav_np.ndim == 2 else wav_np[np.newaxis, :]
    for ch in range(channels):
        result[ch] = np.interp(x_new, x_old, src[ch])
    return result


# ── Timing helpers ─────────────────────────────────────────────────────────────

@dataclass
class TimingBreakdown:
    """Per-call timing breakdown for every pipeline stage (seconds)."""
    model_load_s: float = 0.0       # one-time cost; 0 after first call
    audio_load_s: float = 0.0       # torchaudio decode + resample
    tokenization_s: float = 0.0     # apply_chat_template + processor()
    device_transfer_s: float = 0.0  # .to(device) for input tensors
    generate_s: float = 0.0         # model.generate() — the hot path
    decode_s: float = 0.0           # batch_decode tokens → text
    total_s: float = 0.0            # wall-clock from _run() entry

    def report(self, audio_duration_s: float = 0.0) -> str:
        """Return a human-readable multi-line timing report."""
        rtfx = audio_duration_s / self.total_s if self.total_s > 0 else 0.0
        pct = lambda v: f"{100*v/self.total_s:.1f}%" if self.total_s > 0 else "n/a"
        lines = [
            "┌─ Timing breakdown ──────────────────────────────────",
            f"│  model load       {self.model_load_s:7.3f} s  {pct(self.model_load_s)}",
            f"│  audio load       {self.audio_load_s:7.3f} s  {pct(self.audio_load_s)}",
            f"│  tokenisation     {self.tokenization_s:7.3f} s  {pct(self.tokenization_s)}",
            f"│  device transfer  {self.device_transfer_s:7.3f} s  {pct(self.device_transfer_s)}",
            f"│  generate()       {self.generate_s:7.3f} s  {pct(self.generate_s)}",
            f"│  decode tokens    {self.decode_s:7.3f} s  {pct(self.decode_s)}",
            f"├─────────────────────────────────────────────────────",
            f"│  TOTAL            {self.total_s:7.3f} s",
        ]
        if audio_duration_s > 0:
            lines.append(f"│  audio duration   {audio_duration_s:7.3f} s")
            lines.append(f"│  RTFx             {rtfx:7.1f}×  (>1 = faster than real-time)")
        lines.append("└─────────────────────────────────────────────────────")
        return "\n".join(lines)


@contextmanager
def _timer(store: dict, key: str):
    """Context manager that accumulates elapsed seconds into store[key]."""
    t0 = time.perf_counter()
    yield
    store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class SpeechResult:
    """Container for a single speech-processing result."""
    text: str
    task: str                   # "transcribe" | "translate_to_en" | …
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    audio_duration_s: float = 0.0
    processing_time_s: float = 0.0
    device_used: str = "cpu"
    model_name: str = MODEL_NAME
    segments: list[dict] = field(default_factory=list)  # per-segment details
    timing: Optional[TimingBreakdown] = None            # detailed profiling

    @property
    def realtime_factor(self) -> float:
        """Higher is faster: audio_duration / processing_time."""
        if self.processing_time_s <= 0:
            return 0.0
        return self.audio_duration_s / self.processing_time_s

    def timing_report(self) -> str:
        """Return full timing breakdown or a minimal summary."""
        if self.timing is not None:
            return self.timing.report(self.audio_duration_s)
        return (
            f"Total: {self.processing_time_s:.3f} s  |  "
            f"RTFx: {self.realtime_factor:.1f}×  |  "
            f"Device: {self.device_used}"
        )


# ── Performance configuration dataclass ────────────────────────────────────────
@dataclass
class PerfConfig:
    """
    Runtime performance knobs.  All options can be set via config/config.yaml
    under [speech.perf] or as environment variables.

    int8_quantize : bool
        Apply torch.quantization.quantize_dynamic (INT-8) to the LLM backbone
        after loading.  Gives ~2× CPU speedup at minor accuracy cost.
        Has no effect on CUDA/MPS (BitsAndBytes INT-8 is separate).

    compile : bool
        Call torch.compile() on the model (PyTorch ≥ 2.0).  Adds a one-time
        compile cost (~60 s) then speeds up every subsequent call.

    use_cache : bool
        Enable KV-cache in generate().  Default True; set False only for
        memory-constrained devices.

    max_new_tokens_short : int
        Override max_new_tokens for clips ≤ short_clip_threshold_s seconds.
        Prevents over-generating for short inputs (the main CPU bottleneck).

    short_clip_threshold_s : float
        Clips at or below this duration use max_new_tokens_short.

    num_beams : int
        Beam search width.  1 = greedy (fastest).  >1 = better accuracy, slower.

    profile : bool
        Emit the full TimingBreakdown into SpeechResult.timing.
    """
    int8_quantize: bool = False
    compile: bool = False
    use_cache: bool = True
    max_new_tokens_short: int = 64
    short_clip_threshold_s: float = 10.0
    num_beams: int = 1
    profile: bool = True        # always-on by default for observability


# ── Processor class ────────────────────────────────────────────────────────────
class GraniteSpeechProcessor:
    """
    Wraps IBM Granite Speech 5.0-2B with per-stage timing instrumentation and
    configurable performance optimisations.

    Parameters
    ----------
    device : str | None
        Explicit device string ("cuda", "mps", "cpu").  If None, auto-detected.
    torch_dtype : torch.dtype
        Weight precision.  bfloat16 (CUDA), float16 (MPS), float32 (CPU).
    model_name : str
        HuggingFace model identifier.
    cache_dir : str | None
        Local cache dir for HF weights.
    perf : PerfConfig | None
        Performance flags.  Defaults to PerfConfig() (profiling on, rest off).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_name: str = MODEL_NAME,
        cache_dir: Optional[str] = None,
        perf: Optional[PerfConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.perf = perf or PerfConfig()
        self._device: Optional[str] = device
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_load_s: float = 0.0  # recorded once at first load

    # ── public ────────────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        return self._device or self._default_device()

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Pre-load the model (optional; also triggered on first transcribe/translate)."""
        self._load_model()

    def unload(self) -> None:
        """Release model weights from memory / VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Granite Speech model unloaded.")

    def transcribe(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        language: str = "en",
        verbatim: bool = False,
        max_new_tokens: int = 512,
    ) -> SpeechResult:
        """
        Transcribe speech audio to text.

        Parameters
        ----------
        audio : file path, numpy float32 array (16 kHz mono), or torch.Tensor
        language : BCP-47 source language code
        verbatim : use verbatim prompt (no punctuation normalisation)
        max_new_tokens : max tokens per segment (overridden by perf.max_new_tokens_short
                         for short clips when perf.short_clip_threshold_s applies)
        """
        task = "transcribe_verbatim" if verbatim else "transcribe"
        prompt = PROMPT_TEMPLATES[task]
        return self._run(audio, prompt, task, language, None, max_new_tokens)

    def translate(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        source_language: str,
        target_language: str = "en",
        max_new_tokens: int = 512,
    ) -> SpeechResult:
        """
        Translate speech audio (any supported lang → English, or English → others).

        ``source_language`` and ``target_language`` accept BCP-47 codes (e.g. "fr")
        or full names (e.g. "French").  The model prompt **requires** the full name —
        passing a BCP-47 code like "fr" causes the model to ignore the target and
        return the source language instead.
        """
        if target_language == "en":
            prompt = PROMPT_TEMPLATES["translate_to_en"]
            task = "translate_to_en"
        else:
            # Resolve BCP-47 code → full name for the prompt.
            # e.g. "fr" → "French"; unknown codes pass through unchanged.
            target_name = LANGUAGE_NAMES.get(target_language, target_language)
            prompt = PROMPT_TEMPLATES["translate_from_en"].format(
                target_language=target_name
            )
            task = "translate_from_en"
            logger.debug(
                "AST prompt: target_language=%r → full name=%r → prompt=%r",
                target_language, target_name, prompt,
            )
        return self._run(
            audio, prompt, task, source_language, target_language, max_new_tokens
        )

    # ── internals ──────────────────────────────────────────────────────────────
    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Lazy-load weights; idempotent.  Records load time for profiling."""
        if self._model is not None:
            return

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore

        resolved_device = self._device or self._default_device()
        self._device = resolved_device

        logger.info(
            "Loading %s on %s with dtype=%s …",
            self.model_name, resolved_device, self.torch_dtype,
        )
        t_load_start = time.perf_counter()

        # ── Choose device_map strategy ────────────────────────────────────────
        # device_map= dispatches through accelerate; only use it for strategy
        # tokens ("auto", "balanced") — not for plain device strings like "cpu".
        kwargs: dict = {"torch_dtype": self.torch_dtype}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        _accelerate_ok = False
        try:
            import accelerate  # type: ignore  # noqa: F401
            _accelerate_ok = True
        except ImportError:
            pass

        if _accelerate_ok and resolved_device in ("auto", "balanced"):
            kwargs["device_map"] = resolved_device
        # else: plain .to(device) after loading avoids the accelerate requirement

        # ── Load processor + model ────────────────────────────────────────────
        proc_kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        self._processor = AutoProcessor.from_pretrained(self.model_name, **proc_kwargs)
        self._tokenizer = self._processor.tokenizer

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, **kwargs
        )

        if "device_map" not in kwargs:
            self._model = self._model.to(resolved_device)

        self._model.eval()

        # ── Optional INT-8 dynamic quantisation (CPU only) ────────────────────
        # Reduces memory by ~50 % and speeds up CPU inference ~2×.
        # Applied after .to() so the model is on CPU before quantisation.
        if self.perf.int8_quantize and resolved_device == "cpu":
            logger.info("Applying INT-8 dynamic quantisation (CPU) …")
            import torch.quantization  # type: ignore
            # Quantise Linear layers in the LLM backbone only (not the audio encoder)
            self._model = torch.quantization.quantize_dynamic(
                self._model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            logger.info("INT-8 quantisation applied.")

        # ── Optional torch.compile() (PyTorch ≥ 2.0) ─────────────────────────
        # One-time compile cost ~60 s; subsequent calls ~2-4× faster on CPU.
        if self.perf.compile:
            try:
                logger.info("Compiling model with torch.compile() — may take ~60 s …")
                self._model = torch.compile(self._model)
                logger.info("torch.compile() done.")
            except Exception as exc:
                logger.warning("torch.compile() failed (non-fatal): %s", exc)

        self._model_load_s = time.perf_counter() - t_load_start
        logger.info("Model loaded in %.1f s.", self._model_load_s)

    def _load_audio(
        self, audio: str | Path | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Normalise any audio input to a 1-D float32 16 kHz torch.Tensor.
        Returns (waveform, duration_seconds).

        File loading uses :func:`_load_audio_file` which tries multiple audio
        backends in order (torchaudio<2.9 → soundfile → scipy → torchcodec)
        so the app is not broken by the torchaudio≥2.9 backend removal.
        Resampling uses :func:`_resample_numpy` with the same fallback chain.
        """
        if isinstance(audio, (str, Path)):
            # _load_audio_file returns (channels, samples) float32 numpy array
            wav_np, sr = _load_audio_file(str(audio))
            if sr != TARGET_SAMPLE_RATE:
                wav_np = _resample_numpy(wav_np, sr, TARGET_SAMPLE_RATE)
            # Down-mix to mono: average across channels axis
            if wav_np.shape[0] > 1:
                wav_np = wav_np.mean(axis=0, keepdims=True)
            wav = torch.from_numpy(wav_np)
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio.astype(np.float32))
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
        else:
            wav = audio.float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)

        duration = wav.shape[-1] / TARGET_SAMPLE_RATE
        return wav.squeeze(0), duration   # 1-D tensor

    def _split_segments(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """
        Split audio into ≤ MAX_SEGMENT_SECONDS segments (hard-frame-aligned).
        For production use, apply VAD before this step.
        """
        max_samples = MAX_SEGMENT_SECONDS * TARGET_SAMPLE_RATE
        n_samples = waveform.shape[0]
        if n_samples <= max_samples:
            return [waveform]
        return [
            waveform[start: start + max_samples]
            for start in range(0, n_samples, max_samples)
        ]

    @torch.inference_mode()
    def _decode_segment(
        self,
        waveform: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        timing_store: dict,
    ) -> str:
        """
        Run model forward pass for a single audio segment.
        All stage timings are accumulated into timing_store.
        """
        # ── Tokenisation ──────────────────────────────────────────────────────
        with _timer(timing_store, "tokenization_s"):
            chat = [{"role": "user", "content": prompt}]
            text_input = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=text_input,
                audio=waveform.numpy(),
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )

        # ── Device transfer ───────────────────────────────────────────────────
        with _timer(timing_store, "device_transfer_s"):
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # ── Generation ────────────────────────────────────────────────────────
        # num_beams=1 (greedy) is the fastest path for CPU.
        # use_cache=True keeps KV-cache across tokens (significant speedup).
        with _timer(timing_store, "generate_s"):
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=self.perf.num_beams,
                use_cache=self.perf.use_cache,
            )

        # ── Token decode ──────────────────────────────────────────────────────
        with _timer(timing_store, "decode_s"):
            n_prompt_tokens = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, n_prompt_tokens:]
            text = self._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

        return text

    def _run(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        prompt: str,
        task: str,
        source_language: Optional[str],
        target_language: Optional[str],
        max_new_tokens: int,
    ) -> SpeechResult:
        """Internal dispatch: load model, split audio, decode all segments."""
        t_run_start = time.perf_counter()
        timing_store: dict = {}

        # ── Model load (one-time) ─────────────────────────────────────────────
        with _timer(timing_store, "model_load_s"):
            self._load_model()
        # After first load _model_load_s is recorded inside _load_model;
        # on subsequent calls this timer captures near-zero "already loaded" path.

        # ── Audio load ────────────────────────────────────────────────────────
        with _timer(timing_store, "audio_load_s"):
            waveform, duration = self._load_audio(audio)
            segments = self._split_segments(waveform)

        # ── Adaptive max_new_tokens for short clips ───────────────────────────
        # The generate() call scales with max_new_tokens even on short inputs.
        # Capping it for short clips is the single biggest CPU speedup.
        if duration <= self.perf.short_clip_threshold_s:
            effective_max_tokens = min(max_new_tokens, self.perf.max_new_tokens_short)
            logger.debug(
                "Short clip (%.1f s ≤ %.1f s threshold): "
                "max_new_tokens capped %d → %d",
                duration, self.perf.short_clip_threshold_s,
                max_new_tokens, effective_max_tokens,
            )
        else:
            effective_max_tokens = max_new_tokens

        # ── Decode each segment ───────────────────────────────────────────────
        segment_results: list[dict] = []
        texts: list[str] = []
        for idx, seg in enumerate(segments):
            seg_duration = seg.shape[0] / TARGET_SAMPLE_RATE
            seg_text = self._decode_segment(
                seg, prompt, effective_max_tokens, timing_store
            )
            texts.append(seg_text)
            segment_results.append(
                {"index": idx, "duration_s": seg_duration, "text": seg_text}
            )
            logger.debug("Segment %d/%d: %r", idx + 1, len(segments), seg_text)

        full_text = " ".join(texts)
        total_elapsed = time.perf_counter() - t_run_start

        # ── Build timing breakdown ────────────────────────────────────────────
        breakdown = TimingBreakdown(
            model_load_s=timing_store.get("model_load_s", 0.0),
            audio_load_s=timing_store.get("audio_load_s", 0.0),
            tokenization_s=timing_store.get("tokenization_s", 0.0),
            device_transfer_s=timing_store.get("device_transfer_s", 0.0),
            generate_s=timing_store.get("generate_s", 0.0),
            decode_s=timing_store.get("decode_s", 0.0),
            total_s=total_elapsed,
        ) if self.perf.profile else None

        if breakdown is not None:
            logger.info("\n%s", breakdown.report(duration))

        return SpeechResult(
            text=full_text,
            task=task,
            source_language=source_language,
            target_language=target_language,
            audio_duration_s=duration,
            processing_time_s=total_elapsed,
            device_used=self._device,
            model_name=self.model_name,
            segments=segment_results,
            timing=breakdown,
        )
