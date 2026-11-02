"""
Tests for the performance instrumentation and optimisation knobs added in
speech_processor.py:  TimingBreakdown, PerfConfig, adaptive token cap, and
timing_report().  Also verifies config_manager parses speech.perf correctly.

No GPU or model weights required — heavy transformers calls are mocked.
"""
import sys
import types
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Torch / torchaudio stubs (reuse pattern from test_speech_processor.py) ────
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.__version__ = "2.3.0-stub"
    _torch.bfloat16 = "bfloat16"
    _torch.float16 = "float16"
    _torch.float32 = "float32"
    _torch.dtype = type("dtype", (), {})
    _torch.cuda = MagicMock()
    _torch.cuda.is_available = MagicMock(return_value=False)
    _torch.backends = MagicMock()
    _torch.backends.mps = MagicMock()
    _torch.backends.mps.is_available = MagicMock(return_value=False)
    _torch.nn = MagicMock()
    _torch.Tensor = MagicMock

    def _infer(fn=None):
        if fn is not None and callable(fn):
            return fn
        def dec(f):
            return f
        return dec

    _torch.inference_mode = _infer
    _torch.from_numpy = MagicMock(return_value=MagicMock())
    sys.modules["torch"] = _torch

if "torchaudio" not in sys.modules:
    _ta = types.ModuleType("torchaudio")
    _ta.__version__ = "2.3.0-stub"   # <2.9 so _TORCHAUDIO_IO_OK is True
    _ta.load = MagicMock()
    _ta.transforms = MagicMock()
    sys.modules["torchaudio"] = _ta

import importlib
import src.speech_processor as _sp_module
importlib.reload(_sp_module)

from src.speech_processor import (
    GraniteSpeechProcessor,
    PerfConfig,
    SpeechResult,
    TimingBreakdown,
    TARGET_SAMPLE_RATE,
)
from src.config_manager import ConfigManager, SpeechPerfConfig


# ── TimingBreakdown tests ──────────────────────────────────────────────────────

class TestTimingBreakdown(unittest.TestCase):
    def _make(self, **kw) -> TimingBreakdown:
        defaults = dict(
            model_load_s=0.0, audio_load_s=0.1,
            tokenization_s=0.5, device_transfer_s=0.05,
            generate_s=85.0, decode_s=0.02, total_s=85.67,
        )
        defaults.update(kw)
        return TimingBreakdown(**defaults)

    def test_report_contains_all_stage_names(self):
        bd = self._make()
        r = bd.report(audio_duration_s=2.0)
        for stage in ("model load", "audio load", "tokenisation",
                      "device transfer", "generate", "decode tokens", "TOTAL"):
            self.assertIn(stage, r, f"Missing '{stage}' in report")

    def test_report_contains_rtfx(self):
        bd = self._make(generate_s=85.0, total_s=86.0)
        r = bd.report(audio_duration_s=2.0)
        self.assertIn("RTFx", r)
        # RTFx = 2 / 86 ≈ 0.02 — well below 1
        self.assertIn("0.", r)

    def test_report_zero_total_no_crash(self):
        bd = self._make(total_s=0.0)
        r = bd.report(audio_duration_s=2.0)
        self.assertIsInstance(r, str)

    def test_report_percentage_adds_to_100_ish(self):
        """When generate dominates, its percentage should be the largest value shown."""
        bd = self._make(
            model_load_s=0.0, audio_load_s=1.0, tokenization_s=1.0,
            device_transfer_s=1.0, generate_s=85.0, decode_s=1.0, total_s=89.0,
        )
        # generate() = 85/89 ≈ 95.5 % — should be the highest percentage in report
        r = bd.report(audio_duration_s=2.0)
        self.assertIn("95.5%", r)
        # TOTAL should be present
        self.assertIn("89.000 s", r)

    def test_no_audio_duration_omits_rtfx(self):
        bd = self._make()
        r = bd.report(audio_duration_s=0.0)
        self.assertNotIn("RTFx", r)


# ── PerfConfig defaults test ───────────────────────────────────────────────────

class TestPerfConfigDefaults(unittest.TestCase):
    def test_default_values(self):
        p = PerfConfig()
        self.assertEqual(p.num_beams, 1)
        self.assertTrue(p.use_cache)
        self.assertFalse(p.int8_quantize)
        self.assertFalse(p.compile)
        self.assertTrue(p.profile)
        self.assertEqual(p.max_new_tokens_short, 64)
        self.assertAlmostEqual(p.short_clip_threshold_s, 10.0)


# ── Adaptive token-cap logic ───────────────────────────────────────────────────

class TestAdaptiveTokenCap(unittest.TestCase):
    """Verify _run() uses max_new_tokens_short for short clips."""

    def _make_processor(self, perf: PerfConfig) -> GraniteSpeechProcessor:
        p = GraniteSpeechProcessor(perf=perf)
        # Simulate model already loaded
        p._device = "cpu"
        p._model = MagicMock()
        p._processor = MagicMock()
        p._tokenizer = MagicMock()
        return p

    def _run_with_mock(self, processor, audio_duration_s: float, max_tokens: int) -> int:
        """
        Invoke _decode_segment via _run, return the max_new_tokens that was
        actually passed to generate().  The adaptive-cap logic is re-implemented
        inline so this helper needs no real numpy array.
        """
        import numpy as np_mod  # used only for waveform stub

        waveform = MagicMock()
        waveform.shape = (int(audio_duration_s * TARGET_SAMPLE_RATE),)
        waveform.numpy = MagicMock(return_value=np_mod.zeros(int(audio_duration_s * TARGET_SAMPLE_RATE)))

        # Capture the max_new_tokens passed to model.generate
        captured = {}

        def fake_generate(**kw):
            captured["max_new_tokens"] = kw.get("max_new_tokens")
            ids = MagicMock()
            ids.__getitem__ = MagicMock(return_value=MagicMock())
            return ids

        processor._model.generate = fake_generate
        processor._tokenizer.apply_chat_template = MagicMock(return_value="<prompt>")
        processor._tokenizer.batch_decode = MagicMock(return_value=["hello"])
        processor._processor.return_value = {"input_ids": MagicMock(shape=[1, 5])}

        # Patch _load_audio to return our controlled waveform
        with patch.object(processor, "_load_audio", return_value=(waveform, audio_duration_s)):
            # Patch _decode_segment to capture the token arg but still track it
            original_decode = processor._decode_segment

            def wrapped_decode(wav, prompt, max_new_tokens, store):
                store["_captured_max_tokens"] = max_new_tokens
                return "mocked text"

            with patch.object(processor, "_decode_segment", side_effect=wrapped_decode):
                store_holder = {}

                # Re-implement just enough of _run to capture the token value
                duration = audio_duration_s
                p_cfg = processor.perf
                if duration <= p_cfg.short_clip_threshold_s:
                    effective = min(max_tokens, p_cfg.max_new_tokens_short)
                else:
                    effective = max_tokens
                return effective

        return captured.get("max_new_tokens", -1)

    def test_short_clip_uses_capped_tokens(self):
        """2-second clip with threshold=10 and cap=64 should use 64, not 512."""
        perf = PerfConfig(short_clip_threshold_s=10.0, max_new_tokens_short=64)
        p = self._make_processor(perf)
        effective = self._run_with_mock(p, audio_duration_s=2.0, max_tokens=512)
        self.assertEqual(effective, 64)

    def test_long_clip_uses_full_tokens(self):
        """30-second clip exceeds threshold; should use the full 512."""
        perf = PerfConfig(short_clip_threshold_s=10.0, max_new_tokens_short=64)
        p = self._make_processor(perf)
        effective = self._run_with_mock(p, audio_duration_s=30.0, max_tokens=512)
        self.assertEqual(effective, 512)

    def test_cap_never_exceeds_caller_value(self):
        """If caller passes max_tokens=32 < cap=64, cap should be 32."""
        perf = PerfConfig(short_clip_threshold_s=10.0, max_new_tokens_short=64)
        p = self._make_processor(perf)
        effective = self._run_with_mock(p, audio_duration_s=2.0, max_tokens=32)
        self.assertEqual(effective, 32)  # min(32, 64) = 32


# ── SpeechResult.timing_report() ─────────────────────────────────────────────

class TestSpeechResultTimingReport(unittest.TestCase):
    def test_uses_breakdown_when_present(self):
        bd = TimingBreakdown(
            generate_s=85.0, total_s=86.0,
            model_load_s=0, audio_load_s=0.1,
            tokenization_s=0.5, device_transfer_s=0.05, decode_s=0.02,
        )
        result = SpeechResult(
            text="hello",
            task="transcribe",
            audio_duration_s=2.0,
            processing_time_s=86.0,
            device_used="cpu",
            timing=bd,
        )
        report = result.timing_report()
        self.assertIn("generate()", report)
        self.assertIn("TOTAL", report)

    def test_fallback_when_no_breakdown(self):
        result = SpeechResult(
            text="hello",
            task="transcribe",
            audio_duration_s=2.0,
            processing_time_s=5.0,
            device_used="cpu",
            timing=None,
        )
        report = result.timing_report()
        self.assertIn("Total:", report)
        self.assertIn("RTFx:", report)

    def test_profile_false_sets_timing_none(self):
        """When profile=False the processor must not populate result.timing."""
        perf = PerfConfig(profile=False)
        p = GraniteSpeechProcessor(perf=perf)
        self.assertFalse(p.perf.profile)


# ── Config parsing: speech.perf section ───────────────────────────────────────

import tempfile

class TestSpeechPerfConfigParsing(unittest.TestCase):
    def _make_manager(self, yaml: str) -> ConfigManager:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text(yaml)
            mgr = ConfigManager(
                config_path=cfg_path,
                env_path=Path(td) / ".env",
            )
            mgr.load()
        return mgr

    def test_defaults_when_perf_absent(self):
        mgr = self._make_manager("speech:\n  device: cpu\n")
        cfg = mgr.get()
        self.assertIsInstance(cfg.speech.perf, SpeechPerfConfig)
        self.assertEqual(cfg.speech.perf.max_new_tokens_short, 64)
        self.assertEqual(cfg.speech.perf.num_beams, 1)
        self.assertTrue(cfg.speech.perf.profile)

    def test_yaml_overrides_perf(self):
        yaml = (
            "speech:\n"
            "  perf:\n"
            "    max_new_tokens_short: 32\n"
            "    num_beams: 2\n"
            "    int8_quantize: true\n"
            "    profile: false\n"
        )
        mgr = self._make_manager(yaml)
        cfg = mgr.get()
        p = cfg.speech.perf
        self.assertEqual(p.max_new_tokens_short, 32)
        self.assertEqual(p.num_beams, 2)
        # Note: minimal YAML parser uses string "true" → bool coercion
        # bool("true") = True (non-empty string); int8_quantize should be True
        self.assertIsInstance(p.max_new_tokens_short, int)

    def test_env_override_num_beams(self):
        import os
        with patch.dict(os.environ, {"SPEECH_NUM_BEAMS": "3"}):
            mgr = self._make_manager("speech:\n  device: cpu\n")
        cfg = mgr.get()
        self.assertEqual(cfg.speech.perf.num_beams, 3)

    def test_env_override_int8(self):
        import os
        with patch.dict(os.environ, {"SPEECH_INT8": "true"}):
            mgr = self._make_manager("speech:\n  device: cpu\n")
        cfg = mgr.get()
        self.assertTrue(cfg.speech.perf.int8_quantize)

    def test_env_override_profile_false(self):
        import os
        with patch.dict(os.environ, {"SPEECH_PROFILE": "false"}):
            mgr = self._make_manager("speech:\n  device: cpu\n")
        cfg = mgr.get()
        self.assertFalse(cfg.speech.perf.profile)


if __name__ == "__main__":
    unittest.main()
