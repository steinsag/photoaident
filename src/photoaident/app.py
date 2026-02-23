import os
import sys
import threading
from typing import TYPE_CHECKING

import onnxruntime as ort
from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.core.indexer import InventoryTask, IndexingTask
from photoaident.db.database import (
    get_counts,
    clear_database,
    get_engine,
    get_session_factory,
)
from photoaident.db.vector_store import VectorStore
from photoaident.settings import Settings
from photoaident.ui.pages.labelling import LabellingPage
from photoaident.ui.pages.library import LibraryPage
from photoaident.ui.preferences_dialog import PreferencesDialog
from photoaident.ui.widgets.progress_dialog import ProgressDialog

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


class _GPUStatusSignal(QtCore.QObject):
    status_ready = QtCore.Signal(str)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self, paths: "AppPaths", check_gpu: bool = True):
        super().__init__()
        self.paths = paths
        self.settings = Settings.load(self.paths.config_file)

        # Database and vector store
        self.db_engine = get_engine(str(self.paths.db_path))
        self.session_factory = get_session_factory(self.db_engine)
        self.vector_store = VectorStore()
        if self.paths.faiss_path.exists():
            self.vector_store.load(self.paths.faiss_path)

        self.setWindowTitle(self.tr("PhotoAIdent"))
        self.resize(1024, 768)
        self._set_app_icon()

        # Status bar
        self.status_bar = self.statusBar()
        self.counts_label = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.counts_label)
        self.indexing_label = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.indexing_label)

        # Pages
        self.library_page = LibraryPage(
            self.session_factory, self.paths, self.vector_store
        )
        self.labelling_page = LabellingPage(self.session_factory, self.paths)

        # Stacked widget holding the pages
        self.stacked = QtWidgets.QStackedWidget()
        self.stacked.addWidget(self.library_page)  # index 0
        self.stacked.addWidget(self.labelling_page)  # index 1

        # Sidebar navigation buttons
        self.btn_library = QtWidgets.QPushButton(self.tr("Library"))
        self.btn_library.setFlat(True)
        self.btn_library.clicked.connect(lambda: self._switch_page(0))

        self.btn_label = QtWidgets.QPushButton(self.tr("Label"))
        self.btn_label.setFlat(True)
        self.btn_label.clicked.connect(lambda: self._switch_page(1))

        sidebar_layout = QtWidgets.QVBoxLayout()
        sidebar_layout.setContentsMargins(4, 8, 4, 8)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addWidget(self.btn_library)
        sidebar_layout.addWidget(self.btn_label)
        sidebar_layout.addStretch()

        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(110)
        sidebar.setLayout(sidebar_layout)

        # Assemble central widget
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(sidebar)
        central_layout.addWidget(self.stacked)
        self.setCentralWidget(central)

        # Start on Library page
        self._switch_page(0)

        # Create menu bar
        self._create_menus()

        # Update GPU status
        self._gpu_status_signal = _GPUStatusSignal()
        self._gpu_status_signal.status_ready.connect(self._on_gpu_status_ready)
        if check_gpu:
            threading.Thread(target=self._check_gpu, daemon=True).start()

        # Start inventory then indexing at startup
        self._inventory_task: InventoryTask | None = None
        self._inventory_thread: QtCore.QThread | None = None
        self._indexing_task: IndexingTask | None = None
        self._indexing_thread: QtCore.QThread | None = None
        self._update_db_counts()
        QtCore.QTimer.singleShot(1000, self._maybe_start_indexing)

    def _maybe_start_indexing(self) -> None:
        """Run a silent inventory scan then start indexing."""
        if self._indexing_task is not None or self._inventory_task is not None:
            return

        collection_path = self.settings.collection_path
        if not collection_path:
            return

        self.indexing_label.setText(self.tr("Scanning for new photos..."))

        self._inventory_task = InventoryTask(collection_path, self.session_factory)
        self._inventory_thread = QtCore.QThread()
        self._inventory_task.moveToThread(self._inventory_thread)
        self._inventory_task.finished.connect(self._on_startup_inventory_finished)
        self._inventory_thread.started.connect(self._inventory_task.run)
        self._inventory_thread.start()

    def _on_startup_inventory_finished(self, _added: int) -> None:
        """Called when the silent startup inventory scan completes."""
        if self._inventory_thread:
            self._inventory_thread.quit()
            self._inventory_thread.wait()
            self._inventory_thread = None
        self._inventory_task = None
        self._start_indexing_task()

    def _start_indexing_task(self) -> None:
        """Create and start the background indexing task."""
        if self._indexing_task is not None:
            return

        self._indexing_task = IndexingTask(
            self.session_factory, self.vector_store, self.paths
        )
        self._indexing_thread = QtCore.QThread()
        self._indexing_task.moveToThread(self._indexing_thread)

        self._indexing_task.progress.connect(self._update_indexing_status)
        self._indexing_task.finished.connect(self._on_indexing_finished)
        self._indexing_thread.started.connect(self._indexing_task.run)

        self._indexing_thread.start()

    def _update_indexing_status(self, indexed, total, faces):
        msg = self.tr("Indexed: {indexed}/{total} | Faces: {faces}").format(
            indexed=indexed, total=total, faces=faces
        )
        self.indexing_label.setText(msg)
        # Reload library view periodically or when indexing finishes
        # Increased frequency to every 50 images to reduce UI lag
        if indexed % 50 == 0 or indexed == total:
            self.library_page.load_images()

    def _update_db_counts(self) -> None:
        """Refresh the images/faces totals shown in the status bar."""
        image_count, face_count = get_counts(self.session_factory)
        self.counts_label.setText(
            self.tr("Images: {images} | Faces: {faces}").format(
                images=image_count, faces=face_count
            )
        )

    def _on_indexing_finished(self):
        self.indexing_label.setText(self.tr("Indexing complete"))
        if self._indexing_thread:
            self._indexing_thread.quit()
            self._indexing_thread.wait()
            self._indexing_thread = None
        self._indexing_task = None
        self._update_db_counts()
        self.library_page.load_images()

    def _switch_page(self, index: int) -> None:
        """Switch to the given page index and highlight the active sidebar button."""
        buttons = [self.btn_library, self.btn_label]
        for i, btn in enumerate(buttons):
            font = btn.font()
            font.setBold(i == index)
            btn.setFont(font)
        self.stacked.setCurrentIndex(index)
        if index == 1:
            self.labelling_page.refresh()

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
        image_count, face_count = get_counts(self.session_factory)
        old_path = self.settings.collection_path

        dialog = PreferencesDialog(old_path, image_count, face_count, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_path = dialog.get_collection_path()
            if new_path != old_path:
                # Ask for confirmation
                msg = self.tr(
                    "Changing the photo collection path will cause all existing "
                    "detected faces to be lost.\n\n"
                    "Currently indexed:\n"
                    "- {images} images\n"
                    "- {faces} faces\n\n"
                    "Do you really want to proceed?"
                ).format(images=image_count, faces=face_count)

                reply = QtWidgets.QMessageBox.question(
                    self,
                    self.tr("Confirm Collection Change"),
                    msg,
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )

                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    # Clear data
                    clear_database(self.session_factory)
                    self.vector_store.reset()
                    self.vector_store.save(self.paths.faiss_path)

                    # Update settings
                    self.settings.collection_path = new_path
                    self.settings.save(self.paths.config_file)

                    # Start inventory scan
                    self._run_inventory_scan(new_path)
            else:
                # Path didn't change, just save settings
                # (in case other settings added later)
                self.settings.save(self.paths.config_file)

    def _run_inventory_scan(self, path: str):
        """Run the initial inventory scan for a new collection path."""
        dialog = ProgressDialog(
            self.tr("Indexing"), self.tr("Searching for photos..."), self
        )

        task = InventoryTask(path, self.session_factory)
        thread = QtCore.QThread()
        task.moveToThread(thread)

        task.status.connect(dialog.update_status)
        task.progress.connect(dialog.update_progress)
        task.finished.connect(dialog.accept)
        task.finished.connect(thread.quit)
        task.finished.connect(self._start_indexing_task)
        thread.started.connect(task.run)
        thread.finished.connect(thread.deleteLater)

        # Ensure task is deleted when thread finishes
        task.finished.connect(task.deleteLater)

        # Start thread before exec to avoid blocking if finished signal comes early
        thread.start()
        try:
            dialog.exec()
        finally:
            if thread.isRunning():
                task.cancel()
                thread.quit()
                thread.wait()

    def _check_gpu(self):
        try:
            import insightface  # noqa: F401

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

        # Update status bar via signal
        self._gpu_status_signal.status_ready.emit(msg)

    def _on_gpu_status_ready(self, msg: str):
        self.status_bar.showMessage(msg, 5000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Graceful shutdown: cancel any running inventory/indexing tasks
        if self._inventory_task is not None and self._inventory_thread is not None:
            try:
                self._inventory_task.cancel()
                self._inventory_thread.quit()
                self._inventory_thread.wait(3000)
            except Exception:
                pass
        if self._indexing_task is not None and self._indexing_thread is not None:
            try:
                self._indexing_task.cancel()
                # Ask the worker thread to stop and wait a bit
                self._indexing_thread.quit()
                self._indexing_thread.wait(5000)
            except Exception:
                pass
            finally:
                # Ensure FAISS index is persisted
                try:
                    self.vector_store.save(self.paths.faiss_path)
                except Exception:
                    pass
        event.accept()

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
