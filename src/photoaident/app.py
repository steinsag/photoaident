import os
import random
import sys
import threading
from typing import TYPE_CHECKING

import onnxruntime as ort
from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.settings import Settings
from photoaident.ui.preferences_dialog import PreferencesDialog

if TYPE_CHECKING:
    from photoaident.paths import AppPaths


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return os.path.join(sys._MEIPASS, relative_path)

    return os.path.join(os.path.abspath("."), relative_path)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    status_ready = QtCore.Signal(str, str)  # message, color

    def __init__(self, paths: "AppPaths"):
        super().__init__()
        self.paths = paths
        self.settings = Settings.load(self.paths.config_file)

        self.setWindowTitle("PhotoAIdent")
        self.resize(800, 600)
        self._set_app_icon()

        # Central widget
        self.central_widget = MyWidget()
        self.setCentralWidget(self.central_widget)

        # Create menu bar
        self._create_menus()

        # Background thread so UI doesn't freeze during ONNX init
        self.status_ready.connect(self.central_widget._update_status)
        threading.Thread(target=self._check_gpu, daemon=True).start()

    def _create_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        settings_action = QtGui.QAction("&Preferences", self)
        settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        settings_action.triggered.connect(self._show_preferences)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QtGui.QAction("&Exit", self)
        exit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _show_preferences(self):
        """Show the preferences dialog and save changes if accepted."""
        dialog = PreferencesDialog(self.settings.collection_path, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.settings.collection_path = dialog.get_collection_path()
            self.settings.save(self.paths.config_file)

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

    def _set_app_icon(self):
        icon = QtGui.QIcon()
        resolutions = ["512", "256", "128", "64", "48"]
        for res in resolutions:
            path = get_resource_path(f"assets/icons/app-{res}.png")
            if os.path.exists(path):
                icon.addFile(path)

        default_path = get_resource_path("assets/icons/app.png")
        if os.path.exists(default_path):
            icon.addFile(default_path)

        self.setWindowIcon(icon)
        QtWidgets.QApplication.setWindowIcon(icon)


class MyWidget(QtWidgets.QWidget):
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

    @QtCore.Slot(str, str)
    def _update_status(self, message: str, color: str):
        self.gpu_status.setText(message)
        self.gpu_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))
