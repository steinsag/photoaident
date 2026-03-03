import builtins
import sys
import threading
from unittest.mock import MagicMock

import onnxruntime as ort

from photoaident.core.gpu_checker import GpuChecker


def _make_checker() -> GpuChecker:
    return GpuChecker()


def test_probe_emits_gpu_ready_when_cuda_available(monkeypatch):
    """_probe emits a GPU-ready string when CUDAExecutionProvider is present."""
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    if "insightface" not in sys.modules:
        monkeypatch.setitem(sys.modules, "insightface", MagicMock())

    checker = _make_checker()
    captured: list[str] = []
    checker.status_ready.connect(lambda msg: captured.append(msg))

    checker._probe()

    assert len(captured) == 1
    assert "GPU" in captured[0] or "✅" in captured[0]


def test_probe_emits_cpu_only_when_no_cuda(monkeypatch):
    """_probe emits a CPU-only warning when CUDA is not available."""
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    if "insightface" not in sys.modules:
        monkeypatch.setitem(sys.modules, "insightface", MagicMock())

    checker = _make_checker()
    captured: list[str] = []
    checker.status_ready.connect(lambda msg: captured.append(msg))

    checker._probe()

    assert len(captured) == 1
    assert "CPU" in captured[0] or "⚠️" in captured[0]


def test_probe_emits_error_on_import_failure(monkeypatch):
    """_probe emits an error message when insightface cannot be imported."""
    real_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):
        if name == "insightface":
            raise ImportError("mocked: insightface not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "insightface", raising=False)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    checker = _make_checker()
    captured: list[str] = []
    checker.status_ready.connect(lambda msg: captured.append(msg))

    checker._probe()

    assert len(captured) == 1
    assert "❌" in captured[0]


def test_probe_emits_error_on_unexpected_exception(monkeypatch):
    """_probe catches generic exceptions and emits an error string."""
    monkeypatch.setitem(sys.modules, "insightface", MagicMock())
    monkeypatch.setattr(
        ort, "get_available_providers", MagicMock(side_effect=RuntimeError("boom"))
    )

    checker = _make_checker()
    captured: list[str] = []
    checker.status_ready.connect(lambda msg: captured.append(msg))

    checker._probe()

    assert len(captured) == 1
    assert "❌" in captured[0]


def test_start_launches_daemon_thread(monkeypatch):
    """start() spawns a daemon thread that calls _probe."""
    event = threading.Event()
    monkeypatch.setattr(GpuChecker, "_probe", lambda _: event.set())

    checker = _make_checker()
    checker.start()

    assert event.wait(timeout=2.0), "_probe was not called within 2 s"
