"""
Hardware capability detection and optimisation advisor.

Interrogates the host machine for CPU cores, RAM, GPU VRAM and Apple Silicon
presence, then returns a structured HardwareProfile and suggested model-loading
parameters.

No external heavy dependencies – falls back gracefully when optional packages
(torch, psutil) are unavailable so that the module can be imported during the
install phase.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

# Optional torch import at module level so tests can patch it cleanly.
try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    backend: str  # "cuda" | "mps" | "none"

    @property
    def is_usable(self) -> bool:
        """True when free VRAM is likely enough to load the 2B model (~4 GB)."""
        return self.free_vram_gb >= 3.5


@dataclass
class HardwareProfile:
    """
    Snapshot of host hardware relevant to loading Granite Speech 5.0-2B.

    Attributes
    ----------
    cpu_cores_logical : number of logical CPU threads
    cpu_cores_physical : number of physical CPU cores
    ram_total_gb : total system RAM
    ram_available_gb : currently available system RAM
    gpus : list of detected GPU devices
    is_apple_silicon : True when running on Apple Silicon (MPS backend)
    platform : OS identifier string
    python_version : Python version string
    torch_version : torch version string or "not installed"
    recommended_device : "cuda" | "mps" | "cpu"
    recommended_dtype : torch dtype string ("bfloat16" | "float16" | "float32")
    recommended_batch_size : conservative batch size given available memory
    can_load_full_model : True when hardware can likely load the full FP model
    warnings : list of human-readable advisory messages
    """

    cpu_cores_logical: int = 0
    cpu_cores_physical: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpus: list[GPUInfo] = field(default_factory=list)
    is_apple_silicon: bool = False
    platform: str = ""
    python_version: str = ""
    torch_version: str = "not installed"
    recommended_device: str = "cpu"
    recommended_dtype: str = "bfloat16"
    recommended_batch_size: int = 1
    can_load_full_model: bool = False
    warnings: list[str] = field(default_factory=list)

    # ── convenience helpers ────────────────────────────────────────────────────
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        """Return the GPU with the most free VRAM, or None."""
        usable = [g for g in self.gpus if g.is_usable]
        return max(usable, key=lambda g: g.free_vram_gb) if usable else None

    def summary_str(self) -> str:
        """Return a compact human-readable summary string."""
        lines = [
            f"Platform : {self.platform}",
            f"CPU      : {self.cpu_cores_physical}P / {self.cpu_cores_logical}L cores",
            f"RAM      : {self.ram_total_gb:.1f} GB total, "
            f"{self.ram_available_gb:.1f} GB available",
        ]
        for gpu in self.gpus:
            lines.append(
                f"GPU [{gpu.index}] : {gpu.name}  "
                f"{gpu.free_vram_gb:.1f}/{gpu.total_vram_gb:.1f} GB VRAM free"
            )
        if not self.gpus:
            lines.append("GPU      : none detected")
        lines += [
            f"→ device : {self.recommended_device}",
            f"→ dtype  : {self.recommended_dtype}",
            f"→ model  : {'✓ loadable' if self.can_load_full_model else '⚠ may not fit in memory'}",
        ]
        if self.warnings:
            lines += [f"⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


# ── Detector ──────────────────────────────────────────────────────────────────

class HardwareDetector:
    """
    Detect host hardware capabilities and return an HardwareProfile.

    Example
    -------
    >>> profile = HardwareDetector().detect()
    >>> print(profile.summary_str())
    """

    # Approximate RAM needed for the 2B model in various dtypes (GB)
    _MODEL_VRAM_REQUIREMENTS = {
        "bfloat16": 4.0,
        "float16": 4.0,
        "float32": 8.0,
    }

    def detect(self) -> HardwareProfile:
        profile = HardwareProfile()
        profile.platform = f"{platform.system()} {platform.release()} ({platform.machine()})"
        profile.python_version = sys.version.split()[0]

        self._detect_cpu(profile)
        self._detect_ram(profile)
        self._detect_gpu(profile)
        self._make_recommendations(profile)
        return profile

    # ── CPU ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_cpu(profile: HardwareProfile) -> None:
        try:
            import psutil  # type: ignore
            profile.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
            profile.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        except ImportError:
            profile.cpu_cores_logical = os.cpu_count() or 1
            profile.cpu_cores_physical = profile.cpu_cores_logical

    # ── RAM ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_ram(profile: HardwareProfile) -> None:
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            profile.ram_total_gb = vm.total / 1e9
            profile.ram_available_gb = vm.available / 1e9
        except ImportError:
            # Rough fallback – read /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as fh:
                    lines = {
                        line.split(":")[0]: int(line.split()[1])
                        for line in fh
                        if ":" in line
                    }
                profile.ram_total_gb = lines.get("MemTotal", 0) / 1e6
                profile.ram_available_gb = lines.get("MemAvailable", 0) / 1e6
            except Exception:
                profile.warnings.append("Could not determine system RAM.")

    # ── GPU ────────────────────────────────────────────────────────────────────
    def _detect_gpu(self, profile: HardwareProfile) -> None:
        # Check Apple Silicon first
        if platform.machine() in ("arm64", "aarch64") and platform.system() == "Darwin":
            self._detect_mps(profile)
            return

        # CUDA devices
        self._detect_cuda(profile)

    @staticmethod
    def _detect_mps(profile: HardwareProfile) -> None:
        profile.is_apple_silicon = True
        # MPS does not expose per-device VRAM through Python APIs;
        # approximate from available system RAM (unified memory).
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            available_gb = vm.available / 1e9
        except ImportError:
            available_gb = profile.ram_available_gb

        mps_available = (
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if mps_available:
            profile.gpus.append(
                GPUInfo(
                    index=0,
                    name="Apple Silicon (MPS)",
                    total_vram_gb=profile.ram_total_gb,  # unified memory
                    free_vram_gb=available_gb,
                    backend="mps",
                )
            )
            logger.info("Apple Silicon MPS backend detected.")

    @staticmethod
    def _detect_cuda(profile: HardwareProfile) -> None:
        if torch is None:
            profile.warnings.append("torch not installed; GPU detection skipped.")
            return
        try:
            if not torch.cuda.is_available():
                return
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                total_gb = props.total_memory / 1e9
                # Snapshot free memory via reserved/allocated counters
                reserved = torch.cuda.memory_reserved(idx)
                allocated = torch.cuda.memory_allocated(idx)
                free_gb = (props.total_memory - max(reserved, allocated)) / 1e9
                profile.gpus.append(
                    GPUInfo(
                        index=idx,
                        name=props.name,
                        total_vram_gb=total_gb,
                        free_vram_gb=free_gb,
                        backend="cuda",
                    )
                )
                logger.info(
                    "CUDA GPU [%d]: %s  %.1f/%.1f GB free",
                    idx,
                    props.name,
                    free_gb,
                    total_gb,
                )
        except Exception as exc:
            profile.warnings.append(f"GPU detection error: {exc}")

    # ── Torch version ──────────────────────────────────────────────────────────
    @staticmethod
    def _get_torch_version() -> str:
        if torch is None:
            return "not installed"
        return getattr(torch, "__version__", "unknown")

    # ── Recommendations ────────────────────────────────────────────────────────
    def _make_recommendations(self, profile: HardwareProfile) -> None:
        profile.torch_version = self._get_torch_version()

        # Device selection
        best_gpu = profile.best_gpu
        if best_gpu is not None:
            profile.recommended_device = best_gpu.backend
        else:
            profile.recommended_device = "cpu"
            if not profile.gpus:
                profile.warnings.append(
                    "No GPU detected; CPU inference will be slow (expect ≥ 10× real-time)."
                )

        # Dtype selection
        if profile.recommended_device == "cpu":
            # bfloat16 is supported on modern CPUs (AVX-512 BF16) but float32 is safer
            profile.recommended_dtype = "float32"
            profile.warnings.append(
                "CPU device selected – using float32 for maximum compatibility. "
                "Model may require ~8 GB RAM."
            )
        elif profile.recommended_device == "mps":
            # MPS supports float16 but not bfloat16 for all ops; use float16
            profile.recommended_dtype = "float16"
        else:
            profile.recommended_dtype = "bfloat16"  # bfloat16 on CUDA

        # Memory feasibility
        req_gb = self._MODEL_VRAM_REQUIREMENTS.get(profile.recommended_dtype, 4.0)
        if best_gpu is not None:
            profile.can_load_full_model = best_gpu.free_vram_gb >= req_gb
        else:
            profile.can_load_full_model = profile.ram_available_gb >= req_gb

        if not profile.can_load_full_model:
            avail = (
                best_gpu.free_vram_gb if best_gpu else profile.ram_available_gb
            )
            profile.warnings.append(
                f"Available memory ({avail:.1f} GB) may be insufficient to load the model "
                f"({req_gb:.1f} GB required for {profile.recommended_dtype}). "
                "Consider reducing dtype precision or using a quantised GGUF variant."
            )

        # Batch size – conservative: 1 for all cases given audio is typically streamed
        profile.recommended_batch_size = 1


# ── Module-level convenience function ─────────────────────────────────────────

def detect_hardware() -> HardwareProfile:
    """
    Detect host hardware and return an HardwareProfile.

    Usage
    -----
    >>> from src.hardware_detector import detect_hardware
    >>> profile = detect_hardware()
    >>> print(profile.summary_str())
    """
    return HardwareDetector().detect()
