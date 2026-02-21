import random
import threading

import onnxruntime as ort
from PySide6 import QtCore, QtWidgets


class MyWidget(QtWidgets.QWidget):
    status_ready = QtCore.Signal(str, str)  # message, color

    def __init__(self):
        super().__init__()

        self.hello = [
            "Hallo Welt",
            "Hei maailma",
            "Hola Mundo",
            "Привіт, світе!",
            "Hello World!",
        ]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel(
            "Hello World", alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.gpu_status = QtWidgets.QLabel(
            "⏳ Checking GPU / InsightFace...",
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        self.gpu_status.setStyleSheet("color: gray; font-size: 11px;")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.text)
        main_layout.addWidget(self.button)
        main_layout.addWidget(self.gpu_status)

        self.button.clicked.connect(self.magic)
        self.status_ready.connect(self._update_status)

        # Background thread so UI doesn't freeze during ONNX init
        threading.Thread(target=self._check_gpu, daemon=True).start()

    def _check_gpu(self):
        try:
            import insightface  # noqa: F401

            providers = ort.get_available_providers()  # type: ignore[attr-defined]
            has_cuda = "CUDAExecutionProvider" in providers

            if has_cuda:
                msg = f"✅ GPU ready — {', '.join(providers)}"
                color = "green"
            else:
                msg = f"⚠️ CPU only — {', '.join(providers)}"
                color = "orange"

        except ImportError as e:
            msg = f"❌ Import failed: {e}"
            color = "red"
        except Exception as e:
            msg = f"❌ Error: {e}"
            color = "red"

        self.status_ready.emit(msg, color)

    @QtCore.Slot(str, str)
    def _update_status(self, message: str, color: str):
        self.gpu_status.setText(message)
        self.gpu_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))
