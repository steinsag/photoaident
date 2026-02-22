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

    # In development, resources are in project_root/assets.
    # project_root is two levels up from this file (src/photoaident/app.py)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, relative_path)


def load_translations(app: QtWidgets.QApplication):
    """Load translations for the current system locale."""
    locale = QtCore.QLocale.system().name()  # e.g. "en_US", "de_DE"

    # We also check the base name (e.g. "de" for "de_DE")
    short_locale = locale.split("_")[0]

    translator = QtCore.QTranslator(app)

    # Search paths for translation files
    # 1. assets/translations/photoaident_<locale>.qm
    # 2. assets/translations/photoaident_<short_locale>.qm

    search_locales = [locale, short_locale]
    for loc in search_locales:
        filename = f"photoaident_{loc}.qm"
        path = get_resource_path(os.path.join("assets", "translations", filename))
        if os.path.exists(path):
            if translator.load(path):
                app.installTranslator(translator)
                # Keep a reference to prevent garbage collection
                app._translator = translator  # type: ignore[attr-defined]
                break


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    status_ready = QtCore.Signal(str, str)  # message, color

    def __init__(self, paths: "AppPaths"):
        super().__init__()
        self.paths = paths
        self.settings = Settings.load(self.paths.config_file)

        self.setWindowTitle(self.tr("PhotoAIdent"))
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
        file_menu = menubar.addMenu(self.tr("&File"))

        settings_action = QtGui.QAction(self.tr("&Preferences"), self)
        settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        settings_action.triggered.connect(self._show_preferences)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QtGui.QAction(self.tr("&Exit"), self)
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
                msg = self.tr("✅ GPU ready — {providers}").format(
                    providers=", ".join(providers)
                )
                color = "green"
            else:
                msg = self.tr("⚠️ CPU only — {providers}").format(
                    providers=", ".join(providers)
                )
                color = "orange"

        except ImportError as e:
            msg = self.tr("❌ Import failed: {error}").format(error=str(e))
            color = "red"
        except Exception as e:
            msg = self.tr("❌ Error: {error}").format(error=str(e))
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
            self.tr("Hallo Welt"),
            self.tr("Hei maailma"),
            self.tr("Hola Mundo"),
            self.tr("Привіт, світе!"),
            self.tr("Hello World!"),
        ]

        self.button = QtWidgets.QPushButton(self.tr("Click me!"))
        self.text = QtWidgets.QLabel(
            self.tr("Hello World"), alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.gpu_status = QtWidgets.QLabel(
            self.tr("⏳ Checking GPU / InsightFace..."),
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
