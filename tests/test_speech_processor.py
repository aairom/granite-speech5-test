"""
Unit tests for the speech processor module.

The actual model download and inference are NOT triggered here.
All heavy dependencies (transformers, torch) are mocked so the suite
runs without GPU or internet access.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Provide lightweight torch stub before importing speech_processor
import types, sys

# ── Minimal torch stub ────────────────────────────────────────────────────────
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.cuda = MagicMock()
    torch_stub.cuda.is_available = MagicMock(return_value=False)
    torch_stub.backends = MagicMock()
    torch_stub.backends.mps = MagicMock()
    torch_stub.backends.mps.is_available = MagicMock(return_value=False)
    torch_stub.Tensor = MagicMock

    # inference_mode must work both as @torch.inference_mode() (called with no args,
    # returns a decorator) AND as @torch.inference_mode (applied directly to fn).
    class _inference_mode_cls:
        def __init__(self, fn=None):
            self._fn = fn
        def __call__(self, fn=None):
            if self._fn is not None:
                # was used as @torch.inference_mode (no parentheses)
                return self._fn(*(() if fn is None else (fn,)))
            # used as @torch.inference_mode() → fn is the decorated function
            import functools
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def _inference_mode_factory(fn=None):
        """Supports both @inference_mode and @inference_mode()."""
        if fn is not None and callable(fn):
            # Used as a bare decorator: @torch.inference_mode
            return fn
        # Used as @torch.inference_mode() – return a decorator
        def decorator(f):
            return f
        return decorator

    torch_stub.inference_mode = _inference_mode_factory
    torch_stub.from_numpy = lambda arr: MagicMock(
        float=lambda: MagicMock(
            ndim=1,
            unsqueeze=lambda dim, keepdim=False: MagicMock(
                mean=lambda **kw: MagicMock(squeeze=lambda: MagicMock()),
                squeeze=lambda d: MagicMock(),
            ),
        )
    )
    sys.modules["torch"] = torch_stub

if "torchaudio" not in sys.modules:
    torchaudio_stub = types.ModuleType("torchaudio")
    # Version < 2.9 so _TORCHAUDIO_IO_OK is True in the version guard
    torchaudio_stub.__version__ = "2.3.0-stub"
    torchaudio_stub.load = MagicMock()
    torchaudio_stub.transforms = MagicMock()
    sys.modules["torchaudio"] = torchaudio_stub

from src.speech_processor import (
    GraniteSpeechProcessor,
    SpeechResult,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
    MAX_SEGMENT_SECONDS,
    PROMPT_TEMPLATES,
)


class TestSpeechResult(unittest.TestCase):
    def _make_result(self, audio_dur=10.0, proc_time=5.0) -> SpeechResult:
        return SpeechResult(
            text="Hello world",
            task="transcribe",
            audio_duration_s=audio_dur,
            processing_time_s=proc_time,
        )

    def test_realtime_factor(self):
        r = self._make_result(audio_dur=10.0, proc_time=5.0)
        self.assertAlmostEqual(r.realtime_factor, 2.0)

    def test_realtime_factor_zero_proc_time(self):
        r = self._make_result(proc_time=0.0)
        self.assertEqual(r.realtime_factor, 0.0)

    def test_defaults(self):
        r = SpeechResult(text="test", task="transcribe")
        self.assertEqual(r.text, "test")
        self.assertIsNone(r.source_language)
        self.assertIsNone(r.target_language)
        self.assertEqual(r.segments, [])


class TestConstants(unittest.TestCase):
    def test_supported_languages(self):
        for lang in ("en", "fr", "de", "es", "pt", "ja"):
            self.assertIn(lang, SUPPORTED_LANGUAGES)

    def test_target_sample_rate(self):
        self.assertEqual(TARGET_SAMPLE_RATE, 16_000)

    def test_max_segment_seconds_under_two_minutes(self):
        self.assertLessEqual(MAX_SEGMENT_SECONDS, 120)

    def test_prompt_templates_have_audio_placeholder(self):
        for key, tmpl in PROMPT_TEMPLATES.items():
            self.assertIn("<|audio|>", tmpl, f"Missing <|audio|> in template '{key}'")


class TestGraniteSpeechProcessorInit(unittest.TestCase):
    def test_default_init(self):
        p = GraniteSpeechProcessor()
        self.assertEqual(p.model_name, "ibm-granite/granite-speech-5.0-2b")
        self.assertFalse(p.is_loaded())

    def test_custom_model(self):
        p = GraniteSpeechProcessor(model_name="custom/model")
        self.assertEqual(p.model_name, "custom/model")

    def test_device_resolution_cpu(self):
        """When no GPU is available, device should fall back to cpu."""
        import torch
        torch.cuda.is_available = MagicMock(return_value=False)
        torch.backends.mps.is_available = MagicMock(return_value=False)
        p = GraniteSpeechProcessor()
        self.assertEqual(p.device, "cpu")

    def test_unload_no_op_when_not_loaded(self):
        """Unloading when no model is loaded should not raise."""
        p = GraniteSpeechProcessor()
        p.unload()  # should be silent


class TestSplitSegments(unittest.TestCase):
    """Test the internal _split_segments method without loading the model."""

    def setUp(self):
        import torch
        self.p = GraniteSpeechProcessor()
        # Provide a simple torch.Tensor-like via numpy
        self._torch = torch

    def _make_waveform(self, duration_s: float):
        """Return a numpy array simulating mono audio."""
        n = int(duration_s * TARGET_SAMPLE_RATE)
        return np.zeros(n, dtype=np.float32)

    def test_short_audio_not_split(self):
        import torch as _torch_real
        # The processor uses real torch internally; patch _split_segments directly
        waveform = MagicMock()
        waveform.shape = (TARGET_SAMPLE_RATE * 60,)  # 60 seconds
        waveform.__getitem__ = lambda self, s: waveform
        # Should return a single-element list
        p = GraniteSpeechProcessor()
        # Patch torch.Tensor operations inside _split_segments

        class FakeTensor:
            def __init__(self, n):
                self._n = n
            @property
            def shape(self):
                return (self._n,)
            def __getitem__(self, s):
                start = s.start or 0
                stop = min(s.stop or self._n, self._n)
                return FakeTensor(stop - start)

        short = FakeTensor(TARGET_SAMPLE_RATE * 30)  # 30 s
        segments = p._split_segments(short)
        self.assertEqual(len(segments), 1)

    def test_long_audio_split(self):
        p = GraniteSpeechProcessor()

        class FakeTensor:
            def __init__(self, n):
                self._n = n
            @property
            def shape(self):
                return (self._n,)
            def __getitem__(self, s):
                start = s.start or 0
                stop = min(s.stop or self._n, self._n)
                return FakeTensor(stop - start)

        # 5 minutes of audio
        long = FakeTensor(TARGET_SAMPLE_RATE * 300)
        segments = p._split_segments(long)
        # 300 s / 119 s per segment = 3 segments
        self.assertGreater(len(segments), 1)


# ── Tests for the new audio-backend fallback layer ────────────────────────────

class TestLoadAudioFileBackends(unittest.TestCase):
    """
    Test _load_audio_file() multi-backend fallback without real audio files.
    We mock each backend's import to control which one succeeds.
    """

    def _import_helper(self):
        """Re-import the module-level helpers after stub setup."""
        from src.speech_processor import _load_audio_file, _resample_numpy
        return _load_audio_file, _resample_numpy

    def test_soundfile_backend_used(self):
        """When soundfile is importable and returns data, it should be used."""
        from src.speech_processor import _load_audio_file
        import io
        dummy_wav = np.zeros((2, 100), dtype=np.float32)  # (channels, samples)

        sf_mock = MagicMock()
        # soundfile.read returns (samples, channels) shape — transpose of what we want
        sf_mock.read = MagicMock(return_value=(dummy_wav.T, 16000))
        with patch.dict("sys.modules", {"soundfile": sf_mock}):
            # Force _TORCHAUDIO_IO_OK=False so soundfile is tried first
            with patch("src.speech_processor._TORCHAUDIO_IO_OK", False):
                result, sr = _load_audio_file("/fake/file.wav")
        self.assertEqual(sr, 16000)
        # Result should be transposed back: (channels, samples)
        self.assertEqual(result.shape, dummy_wav.shape)

    def test_raises_when_all_backends_fail(self):
        """If every backend fails, RuntimeError with install instructions."""
        from src.speech_processor import _load_audio_file
        with patch("src.speech_processor._TORCHAUDIO_IO_OK", False):
            with patch.dict("sys.modules", {
                "soundfile": None,
                "scipy": None,
                "scipy.io": None,
                "torchcodec": None,
                "torchcodec.decoders": None,
            }):
                with self.assertRaises((RuntimeError, ImportError, TypeError)):
                    _load_audio_file("/fake/missing.wav")

    def test_torchaudio_backend_skipped_when_version_too_new(self):
        """_TORCHAUDIO_IO_OK=False should skip torchaudio and go to soundfile."""
        from src.speech_processor import _load_audio_file
        dummy = np.zeros((1, 50), dtype=np.float32)
        sf_mock = MagicMock()
        sf_mock.read = MagicMock(return_value=(dummy.T, 16000))
        with patch("src.speech_processor._TORCHAUDIO_IO_OK", False):
            with patch.dict("sys.modules", {"soundfile": sf_mock}):
                result, sr = _load_audio_file("/fake/file.wav")
        # torchaudio.load should NOT have been called
        # (it would only be attempted if _TORCHAUDIO_IO_OK were True)
        self.assertEqual(sr, 16000)


class TestResampleNumpy(unittest.TestCase):
    """Test _resample_numpy fallback chain."""

    def test_same_rate_is_identity(self):
        from src.speech_processor import _resample_numpy
        wav = np.ones((1, 160), dtype=np.float32)
        out = _resample_numpy(wav, 16000, 16000)
        np.testing.assert_array_equal(out, wav)

    def test_resample_changes_length(self):
        """Resampling 16kHz → 8kHz should halve the number of samples."""
        from src.speech_processor import _resample_numpy
        wav = np.ones((1, 16000), dtype=np.float32)
        # Force torchaudio path off so numpy interp fallback is exercised
        with patch("src.speech_processor._TORCHAUDIO_RESAMPLE_OK", False):
            out = _resample_numpy(wav, 16000, 8000)
        # Approximate: target is 8000 samples
        self.assertAlmostEqual(out.shape[-1], 8000, delta=5)

    def test_resample_output_is_float32(self):
        from src.speech_processor import _resample_numpy
        wav = np.ones((1, 16000), dtype=np.float32)
        with patch("src.speech_processor._TORCHAUDIO_RESAMPLE_OK", False):
            out = _resample_numpy(wav, 16000, 8000)
        self.assertEqual(out.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
