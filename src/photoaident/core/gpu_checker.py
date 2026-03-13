import threading

import onnxruntime as ort
from PySide6 import QtCore


class GpuChecker(QtCore.QObject):
    """Background GPU/ONNX availability probe.

    Emits ``status_ready`` with a human-readable status string once the probe
    completes. The probe runs in a daemon thread so it never blocks the UI.
    """

    status_ready = QtCore.Signal(str)

    def start(self) -> None:
        """Launch the GPU probe in a daemon thread."""
        threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self) -> None:
        """Check CUDA/ONNX providers and emit ``status_ready``."""
        try:
            __import__("insightface")
            providers = ort.get_available_providers()  # type: ignore
            has_gpu = any(
                p in providers
                for p in (
                    "CUDAExecutionProvider",
                    "CoreMLExecutionProvider",
                    "OpenVINOExecutionProvider",
                )
            )

            prefix = "✅" if has_gpu else "⚠️"
            label = self.tr("GPU/NPU ready") if has_gpu else self.tr("CPU only")
            msg = f"{prefix} {label} — {', '.join(providers)}"

        except Exception as e:
            prefix = (
                self.tr("Import failed")
                if isinstance(e, ImportError)
                else self.tr("Error")
            )
            msg = f"❌ {prefix}: {str(e)}"

        self.status_ready.emit(msg)
