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
            __import__("insightface")  # raises ImportError when not installed
            providers = ort.get_available_providers()  # type: ignore[attr-defined]
            has_cuda = "CUDAExecutionProvider" in providers

            if has_cuda:
                msg = self.tr("✅ GPU ready — {providers}").format(
                    providers=", ".join(providers)
                )
            else:
                msg = self.tr("⚠️ CPU only — {providers}").format(
                    providers=", ".join(providers)
                )

        except ImportError as e:
            msg = self.tr("❌ Import failed: {error}").format(error=str(e))
        except Exception as e:
            msg = self.tr("❌ Error: {error}").format(error=str(e))

        self.status_ready.emit(msg)
