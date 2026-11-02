"""
Unit tests for the hardware detection module.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Inject minimal torch stub so hardware_detector can import without GPU ────
if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.__version__ = "2.3.0-stub"
    # dtype placeholders (used by speech_processor as default arg values)
    _torch_stub.bfloat16 = "bfloat16"
    _torch_stub.float16 = "float16"
    _torch_stub.float32 = "float32"
    _torch_stub.dtype = type("dtype", (), {})  # simple class stand-in
    _torch_stub.cuda = MagicMock()
    _torch_stub.cuda.is_available = MagicMock(return_value=False)
    _torch_stub.cuda.device_count = MagicMock(return_value=0)
    _torch_stub.backends = MagicMock()
    _torch_stub.backends.mps = MagicMock()
    _torch_stub.backends.mps.is_available = MagicMock(return_value=False)

    def _inference_mode_factory(fn=None):
        if fn is not None and callable(fn):
            return fn
        def decorator(f):
            return f
        return decorator

    _torch_stub.inference_mode = _inference_mode_factory
    _torch_stub.Tensor = MagicMock
    _torch_stub.from_numpy = MagicMock(return_value=MagicMock())
    sys.modules["torch"] = _torch_stub

# Stub torchaudio too so speech_processor doesn't fail at import
if "torchaudio" not in sys.modules:
    _ta_stub = types.ModuleType("torchaudio")
    _ta_stub.__version__ = "2.3.0-stub"  # <2.9 so _TORCHAUDIO_IO_OK is True
    _ta_stub.load = MagicMock()
    _ta_stub.transforms = MagicMock()
    sys.modules["torchaudio"] = _ta_stub

# Re-import hardware_detector AFTER injecting the stub so it picks up the stub.
import importlib
import src.hardware_detector as _hd_module
importlib.reload(_hd_module)

from src.hardware_detector import (
    GPUInfo,
    HardwareDetector,
    HardwareProfile,
    detect_hardware,
)


class TestGPUInfo(unittest.TestCase):
    def test_is_usable_true(self):
        gpu = GPUInfo(index=0, name="RTX 4090", total_vram_gb=24.0, free_vram_gb=20.0, backend="cuda")
        self.assertTrue(gpu.is_usable)

    def test_is_usable_false_low_vram(self):
        gpu = GPUInfo(index=0, name="GTX 1060", total_vram_gb=6.0, free_vram_gb=2.0, backend="cuda")
        self.assertFalse(gpu.is_usable)

    def test_is_usable_threshold(self):
        gpu = GPUInfo(index=0, name="GPU", total_vram_gb=8.0, free_vram_gb=3.5, backend="cuda")
        self.assertTrue(gpu.is_usable)


class TestHardwareProfile(unittest.TestCase):
    def test_best_gpu_no_gpus(self):
        profile = HardwareProfile()
        self.assertIsNone(profile.best_gpu)

    def test_best_gpu_selects_most_free_vram(self):
        profile = HardwareProfile(gpus=[
            GPUInfo(0, "GPU-A", 8.0, 4.0, "cuda"),
            GPUInfo(1, "GPU-B", 16.0, 12.0, "cuda"),
        ])
        best = profile.best_gpu
        self.assertIsNotNone(best)
        self.assertEqual(best.name, "GPU-B")

    def test_best_gpu_ignores_unusable(self):
        profile = HardwareProfile(gpus=[
            GPUInfo(0, "GPU-A", 8.0, 1.0, "cuda"),   # unusable
            GPUInfo(1, "GPU-B", 8.0, 2.0, "cuda"),   # unusable
        ])
        self.assertIsNone(profile.best_gpu)

    def test_summary_str_contains_keywords(self):
        profile = HardwareProfile(
            cpu_cores_physical=8,
            cpu_cores_logical=16,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            recommended_device="cuda",
            recommended_dtype="bfloat16",
            can_load_full_model=True,
        )
        summary = profile.summary_str()
        self.assertIn("cuda", summary)
        self.assertIn("bfloat16", summary)
        self.assertIn("✓", summary)


class TestHardwareDetector(unittest.TestCase):
    def _make_detector(self) -> HardwareDetector:
        return HardwareDetector()

    def test_detect_returns_profile(self):
        """detect() should always return a HardwareProfile."""
        detector = self._make_detector()
        profile = detector.detect()
        self.assertIsInstance(profile, HardwareProfile)
        self.assertGreater(profile.cpu_cores_logical, 0)

    def test_cpu_detection(self):
        detector = self._make_detector()
        profile = HardwareProfile()
        detector._detect_cpu(profile)
        self.assertGreater(profile.cpu_cores_logical, 0)
        self.assertGreater(profile.cpu_cores_physical, 0)

    @patch("src.hardware_detector.platform.system", return_value="Darwin")
    @patch("src.hardware_detector.platform.machine", return_value="arm64")
    def test_apple_silicon_detection(self, mock_machine, mock_system):
        detector = self._make_detector()
        profile = HardwareProfile(ram_total_gb=16.0, ram_available_gb=10.0)

        with patch("src.hardware_detector.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = True
            detector._detect_mps(profile)

        self.assertTrue(profile.is_apple_silicon)
        self.assertEqual(len(profile.gpus), 1)
        self.assertEqual(profile.gpus[0].backend, "mps")

    def test_recommendations_cpu_fallback(self):
        detector = self._make_detector()
        profile = HardwareProfile(
            ram_total_gb=16.0, ram_available_gb=10.0, gpus=[]
        )
        detector._make_recommendations(profile)
        self.assertEqual(profile.recommended_device, "cpu")
        self.assertIn("CPU", " ".join(profile.warnings) + "cpu")

    def test_recommendations_cuda(self):
        detector = self._make_detector()
        profile = HardwareProfile(
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpus=[GPUInfo(0, "RTX 4090", 24.0, 20.0, "cuda")],
        )
        with patch.object(detector, "_get_torch_version", return_value="2.3.0"):
            detector._make_recommendations(profile)
        self.assertEqual(profile.recommended_device, "cuda")
        self.assertEqual(profile.recommended_dtype, "bfloat16")
        self.assertTrue(profile.can_load_full_model)

    def test_recommendations_insufficient_vram_warning(self):
        detector = self._make_detector()
        profile = HardwareProfile(
            ram_total_gb=8.0,
            ram_available_gb=2.0,
            gpus=[GPUInfo(0, "Old GPU", 4.0, 1.5, "cuda")],  # unusable
        )
        with patch.object(detector, "_get_torch_version", return_value="2.3.0"):
            detector._make_recommendations(profile)
        self.assertFalse(profile.can_load_full_model)
        self.assertTrue(len(profile.warnings) > 0)

    def test_module_level_detect_hardware(self):
        profile = detect_hardware()
        self.assertIsInstance(profile, HardwareProfile)


if __name__ == "__main__":
    unittest.main()
