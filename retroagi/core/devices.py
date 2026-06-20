"""Runtime device selection helpers for CPU, CUDA, and Apple Silicon."""

from __future__ import annotations

from typing import Optional

import torch


_MPS_ALIASES = {
    "apple",
    "apple-silicon",
    "apple_silicon",
    "metal",
}


def is_mps_built() -> bool:
    """Return whether the installed PyTorch build includes Apple MPS support."""
    mps_backend = getattr(torch.backends, "mps", None)
    is_built = getattr(mps_backend, "is_built", None)
    return bool(callable(is_built) and is_built())


def is_mps_available() -> bool:
    """Return whether Apple MPS is built and available on this machine."""
    mps_backend = getattr(torch.backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    return bool(callable(is_available) and is_available())


def _normalize_device_name(name: Optional[str | torch.device]) -> str:
    if name is None:
        return "auto"
    normalized = str(name).strip().lower()
    if not normalized:
        return "auto"
    if normalized in _MPS_ALIASES:
        return "mps"
    return normalized


def select_device(name: Optional[str | torch.device] = "auto") -> torch.device:
    """Resolve a user device choice into a validated ``torch.device``.

    ``auto`` prefers CUDA, then Apple Silicon MPS, then CPU. Explicit accelerator
    choices fail early when the requested backend is unavailable.
    """
    normalized = _normalize_device_name(name)

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(normalized)
    if device.type == "cpu":
        return device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
        return device
    if device.type == "mps":
        if not is_mps_built():
            raise RuntimeError("Apple Silicon MPS was requested, but this PyTorch build lacks MPS")
        if not is_mps_available():
            raise RuntimeError("Apple Silicon MPS was requested, but it is unavailable on this machine")
        return device

    raise ValueError(
        f"Unsupported device {name!r}; expected one of auto, cpu, cuda, mps, or apple-silicon"
    )
