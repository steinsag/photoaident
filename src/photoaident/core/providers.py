"""ONNX Runtime execution provider utilities.

Providers are evaluated in preference order:
  TensorRT (NVIDIA) →
  CUDA (NVIDIA) →
  CoreML (macOS/Apple Silicon) →
  OpenVINO (Intel CPU/iGPU/NPU) →
  CPU
"""

import onnxruntime

# Ordered preference: hardware-accelerated providers first, CPU as final fallback.
PREFERRED_PROVIDERS: list[str] = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]

# Providers that indicate hardware acceleration (anything except pure CPU).
HARDWARE_ACCELERATOR_PROVIDERS: frozenset[str] = frozenset(
    p for p in PREFERRED_PROVIDERS if p != "CPUExecutionProvider"
)


def select_providers() -> list[str]:
    """Return available ONNX providers in preferred priority order.

    Always returns at least ``["CPUExecutionProvider"]``.
    """
    available = set(onnxruntime.get_available_providers())
    return [p for p in PREFERRED_PROVIDERS if p in available] or [
        "CPUExecutionProvider"
    ]
