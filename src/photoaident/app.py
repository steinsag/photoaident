import logging
import os
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.db.cluster_means import backfill_cluster_means
from photoaident.db.faiss_migration import rebuild_faiss_with_face_ids
from photoaident.core.gpu_checker import GpuChecker
from photoaident.core.indexing_controller import IndexingController
from photoaident.db.database import (
    get_counts,
    clear_database,
    delete_cache_files,
    get_engine,
    get_session_factory,
)
from photoaident.db.vector_store import VectorStore
from photoaident.settings import Settings
from photoaident.utils.resource_path import get_resource_path
from photoaident.ui.about_dialog import AboutDialog
from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry
from photoaident.ui.onboarding_dialog import OnboardingDialog
from photoaident.ui.pages.browse import BrowsePage
from photoaident.ui.pages.labelling import LabellingPage
from photoaident.ui.pages.library import LibraryPage
from photoaident.ui.pages.persons import PersonsPage
from photoaident.ui.preferences_dialog import PreferencesDialog
from photoaident.ui.widgets.progress_dialog import ProgressDialog

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from photoaident.paths import AppPaths


class CorruptIndexError(Exception):
    """Raised at startup when the on-disk FAISS index cannot be read."""


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
        if os.path.exists(path) and translator.load(path):
            app.installTranslator(translator)
            # Keep a reference to prevent garbage collection
            app._translator = translator  # type: ignore[attr-defined]
            # Align QLocale with the loaded UI language so that locale-aware
            # APIs (e.g. QLocale.standaloneMonthName) match the UI language.
            QtCore.QLocale.setDefault(QtCore.QLocale(loc))
            break


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(
        self,
        paths: "AppPaths",
        check_gpu: bool = True,
        enable_onboarding: bool = True,
    ):
        super().__init__()
        self._paths = paths
        self._settings = Settings.load(self._paths.config_file)
        self._onboarding_enabled = enable_onboarding

        # Database and vector store
        self._db_engine = get_engine(str(self._paths.db_path))
        self._session_factory = get_session_factory(self._db_engine)
        self._vector_store = VectorStore()
        faiss_loaded = False
        if self._paths.faiss_path.exists():
            try:
                self._vector_store.load(self._paths.faiss_path)
                faiss_loaded = True
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                raise CorruptIndexError(
                    f"The FAISS index file is corrupt and cannot be loaded.\n\n"
                    f"File: {self._paths.faiss_path}\n\n"
                    f"This can happen when the application was interrupted during "
                    f"indexing. Delete the file and restart to rebuild the index."
                ) from exc
        if faiss_loaded:
            if self._vector_store.needs_migration():
                logger.warning("Migrating FAISS index to database-driven IDs...")
                self._vector_store = rebuild_faiss_with_face_ids(
                    self._vector_store,
                    self._session_factory,
                )
                self._vector_store.save(self._paths.faiss_path)
                logger.warning("FAISS index migration finished.")
            else:
                logger.warning("FAISS index already migrated, no migration needed.")
            backfill_cluster_means(self._session_factory, self._vector_store)

        self.setWindowTitle(self.tr("PhotoAIdent"))

        # First, try to restore only the window geometry. This ensures that a
        # failure in restoring the window state (toolbars/docks, etc.) does not
        # cause us to override a successfully restored geometry with showMaximized().
        geometry_restored = restore_widget_geometry(
            self, self._paths.window_state_file, restore_state=False
        )

        # Then, best-effort restore of the window state. Any failure here is
        # treated as non-fatal for geometry; we ignore the return value.
        restore_widget_geometry(self, self._paths.window_state_file, restore_state=True)

        if not geometry_restored:
            self.showMaximized()

        self._set_app_icon()

        # Status bar
        self._status_bar = self.statusBar()
        self._counts_label = QtWidgets.QLabel()
        self._status_bar.addPermanentWidget(self._counts_label)
        self._indexing_label = QtWidgets.QLabel()
        self._status_bar.addPermanentWidget(self._indexing_label)

        # Pages
        self._library_page = LibraryPage(
            self._session_factory, self._paths, self._vector_store
        )
        self._labelling_page = LabellingPage(
            self._session_factory, self._paths, self._vector_store
        )
        self._persons_page = PersonsPage(
            self._session_factory, self._paths, self._vector_store
        )
        self._browse_page = BrowsePage(
            self._session_factory, self._paths, self._settings, self._vector_store
        )

        # Stacked widget holding the pages (Search=0, Browse=1, Persons=2, Labelling=3)
        self._stacked_pages = QtWidgets.QStackedWidget()
        self._stacked_pages.addWidget(self._library_page)  # index 0 (Search)
        self._stacked_pages.addWidget(self._browse_page)  # index 1
        self._stacked_pages.addWidget(self._persons_page)  # index 2
        self._stacked_pages.addWidget(self._labelling_page)  # index 3

        # Sidebar navigation buttons
        self._page_btn_search = self._make_nav_button(self.tr("Search"), "search.svg")
        self._page_btn_search.clicked.connect(lambda: self._switch_page(0))
        self._page_btn_search.setShortcut(QtGui.QKeySequence("Alt+1"))

        self._page_btn_browse = self._make_nav_button(self.tr("Browse"), "browse.svg")
        self._page_btn_browse.clicked.connect(lambda: self._switch_page(1))
        self._page_btn_browse.setShortcut(QtGui.QKeySequence("Alt+2"))

        self._page_btn_persons = self._make_nav_button(self.tr("Persons"), "person.svg")
        self._page_btn_persons.clicked.connect(lambda: self._switch_page(2))
        self._page_btn_persons.setShortcut(QtGui.QKeySequence("Alt+3"))

        self._page_btn_label = self._make_nav_button(self.tr("Labelling"), "label.svg")
        self._page_btn_label.clicked.connect(lambda: self._switch_page(3))
        self._page_btn_label.setShortcut(QtGui.QKeySequence("Alt+4"))

        nav_group = QtWidgets.QButtonGroup(self)
        nav_group.setExclusive(True)
        for _btn in [
            self._page_btn_search,
            self._page_btn_browse,
            self._page_btn_persons,
            self._page_btn_label,
        ]:
            nav_group.addButton(_btn)

        sidebar_layout = QtWidgets.QVBoxLayout()
        sidebar_layout.setContentsMargins(4, 8, 4, 8)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addWidget(self._page_btn_search)
        sidebar_layout.addWidget(self._page_btn_browse)
        sidebar_layout.addWidget(self._page_btn_persons)
        sidebar_layout.addWidget(self._page_btn_label)
        sidebar_layout.addStretch()

        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(110)
        sidebar.setStyleSheet("""
            QToolButton {
                border: none;
                border-radius: 4px;
                padding: 8px 4px;
            }
            QToolButton:checked {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
            QToolButton:hover:!checked {
                background-color: palette(mid);
            }
            """)
        sidebar.setLayout(sidebar_layout)

        # Vertical separator between sidebar and content
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)

        # Assemble central widget
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(sidebar)
        central_layout.addWidget(separator)
        central_layout.addWidget(self._stacked_pages)
        self.setCentralWidget(central)

        # Start on Search page
        self._switch_page(0)

        # Create menu bar
        self._create_menus()

        # GPU checker
        self._gpu_checker = GpuChecker(self)
        self._gpu_checker.status_ready.connect(self._on_gpu_status_ready)
        if check_gpu:
            self._gpu_checker.start()

        # Indexing controller
        filepath_pattern = (
            self._settings.filepath_date_pattern
            if self._settings.filepath_date_enabled
            else ""
        )
        self._indexing_controller = IndexingController(
            self._session_factory,
            self._vector_store,
            self._paths,
            filepath_date_pattern=filepath_pattern,
            parent=self,
        )
        self._indexing_controller.inventory_progress.connect(
            self._on_inventory_progress
        )
        self._indexing_controller.inventory_finished.connect(
            self._on_inventory_finished
        )
        self._indexing_controller.indexing_progress.connect(
            self._update_indexing_status
        )
        self._indexing_controller.indexing_finished.connect(self._on_indexing_finished)

        self._update_db_counts()
        QtCore.QTimer.singleShot(1000, self._maybe_start_indexing)

    def _show_onboarding(self) -> None:
        """Show the first-run dialog to select the photo collection folder."""
        dialog = OnboardingDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        path = dialog.selected_path()
        if not path:
            return
        self._settings.collection_path = path
        self._settings.filepath_date_enabled = dialog.is_filepath_date_enabled()
        self._settings.filepath_date_pattern = dialog.get_filepath_date_pattern()
        self._settings.save(self._paths.config_file)
        self._indexing_controller.filepath_date_pattern = (
            self._settings.filepath_date_pattern
            if self._settings.filepath_date_enabled
            else ""
        )
        self._run_inventory_scan(path)

    def _maybe_start_indexing(self) -> None:
        """Run a silent inventory scan then start indexing."""
        if self._indexing_controller.is_busy:
            return

        collection_path = self._settings.collection_path
        if not collection_path:
            if self._onboarding_enabled:
                self._show_onboarding()
            return

        self._indexing_label.setText(self.tr("Scanning for new photos..."))
        self._indexing_controller.start_pipeline(collection_path)

    def _update_indexing_status(
        self, indexed: int, total: int, faces: int, status: str | None = None
    ) -> None:
        if status:
            msg = self.tr(
                "{status} | Indexed: {indexed}/{total} | Faces: {faces}"
            ).format(status=status, indexed=indexed, total=total, faces=faces)
        else:
            msg = self.tr("Indexed: {indexed}/{total} | Faces: {faces}").format(
                indexed=indexed, total=total, faces=faces
            )
        self._indexing_label.setText(msg)
        # Reload library view periodically or when indexing finishes
        # Increased frequency to every 50 images to reduce UI lag
        if indexed % 50 == 0 or indexed == total:
            self._library_page.load_images()

    def _update_db_counts(self) -> None:
        """Refresh the images/faces totals shown in the status bar."""
        image_count, face_count = get_counts(self._session_factory)
        self._counts_label.setText(
            self.tr("Images: {images} | Faces: {faces}").format(
                images=image_count, faces=face_count
            )
        )

    def _on_inventory_progress(self, current: int, total: int, status: str) -> None:
        if hasattr(self, "_inventory_dialog"):
            self._inventory_dialog.update_status(status)
            self._inventory_dialog.update_progress(current, total)

    def _on_inventory_finished(self, _: int) -> None:
        if hasattr(self, "_inventory_dialog"):
            self._inventory_dialog.accept()
            del self._inventory_dialog
        self._indexing_controller.start_indexing_only()

    def _on_indexing_finished(self) -> None:
        self._indexing_label.setText(self.tr("Indexing complete"))
        self._update_db_counts()
        self._library_page.load_images()

    @staticmethod
    def _make_nav_button(label: str, icon_name: str) -> QtWidgets.QToolButton:
        """Create a checkable sidebar navigation button with an icon and label."""
        btn = QtWidgets.QToolButton()
        btn.setText(label)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setIconSize(QtCore.QSize(28, 28))
        btn.setCheckable(True)
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        icon_path = get_resource_path(f"assets/icons/{icon_name}")
        btn.setIcon(QtGui.QIcon(icon_path))

        return btn

    def _switch_page(self, index: int) -> None:
        """Switch to the given page index and highlight the active sidebar button."""
        buttons = [
            self._page_btn_search,
            self._page_btn_browse,
            self._page_btn_persons,
            self._page_btn_label,
        ]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)
        self._stacked_pages.setCurrentIndex(index)
        if index == 1:
            self._browse_page.refresh()
        elif index == 2:
            self._persons_page.refresh()
        elif index == 3:
            self._labelling_page.refresh()

    def go_to_labelling(self, priority_image_id: int) -> None:
        """Navigate to the Labelling page, prioritising faces from the given image."""
        buttons = [
            self._page_btn_search,
            self._page_btn_browse,
            self._page_btn_persons,
            self._page_btn_label,
        ]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == 3)
        self._stacked_pages.setCurrentIndex(3)
        self._labelling_page.refresh(priority_image_id=priority_image_id)

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

        # Help menu
        help_menu = menubar.addMenu(self.tr("&Help"))

        about_action = QtGui.QAction(self.tr("&About"), self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self):
        """Show the About dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def _show_preferences(self):
        """Show the preferences dialog and save changes if accepted."""
        image_count, face_count = get_counts(self._session_factory)
        old_path = self._settings.collection_path

        dialog = PreferencesDialog(
            old_path,
            image_count,
            face_count,
            filepath_date_enabled=self._settings.filepath_date_enabled,
            filepath_date_pattern=self._settings.filepath_date_pattern,
            parent=self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_path = dialog.get_collection_path()
            # Always persist filepath date settings immediately.
            self._settings.filepath_date_enabled = dialog.is_filepath_date_enabled()
            self._settings.filepath_date_pattern = dialog.get_filepath_date_pattern()
            # Propagate the updated pattern to the controller so the next
            # indexing run in this session uses the current settings.
            self._indexing_controller.filepath_date_pattern = (
                self._settings.filepath_date_pattern
                if self._settings.filepath_date_enabled
                else ""
            )

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
                    # Clear data — delete cache files before wiping the DB
                    delete_cache_files(
                        self._session_factory,
                        self._paths.face_crops_dir,
                        self._paths.thumbs_dir,
                    )
                    clear_database(self._session_factory)
                    self._vector_store.reset()
                    self._vector_store.save(self._paths.faiss_path)

                    # Update settings
                    self._settings.collection_path = new_path
                    self._settings.save(self._paths.config_file)

                    # Start inventory scan
                    self._run_inventory_scan(new_path)
                else:
                    self._settings.save(self._paths.config_file)
            else:
                self._settings.save(self._paths.config_file)

    def _run_inventory_scan(self, path: str) -> None:
        """Run the initial inventory scan for a new collection path."""
        self._inventory_dialog = ProgressDialog(
            self.tr("Indexing"), self.tr("Searching for photos..."), self
        )
        self._indexing_controller.start_inventory(path)

        try:
            self._inventory_dialog.exec()
        finally:
            if self._indexing_controller.is_busy:
                self._indexing_controller.cancel_inventory()

    def _on_gpu_status_ready(self, msg: str) -> None:
        self._status_bar.showMessage(msg, 5000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        save_widget_geometry(self, self._paths.window_state_file, save_state=True)
        self._indexing_controller.shutdown(self._paths.faiss_path)
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
