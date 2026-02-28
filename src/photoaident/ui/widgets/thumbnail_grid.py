import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from photoaident.db.database import Face, FaceState, Image
from photoaident.ui.widgets.image_detail_dialog import ImageDetailDialog
from photoaident.utils.image_utils import generate_thumbnail

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

PAGE_SIZE = 30
_SCROLL_LOAD_MARGIN = 200  # px from bottom triggers load
_THUMBNAIL_MAX_SIZE = 150  # max dimension for thumbnail scaling
_THUMBNAIL_WIDGET_SIZE = 160  # fixed widget/overlay size (px)
_THUMBNAIL_GRID_SPACING = 10  # spacing between thumbnail widgets in the grid
_SCROLL_AREA_MARGIN = 20  # reserved for scrollbar in column width calc


def _icon_path(name: str) -> str:
    """Return the path to an icon, works for dev and PyInstaller bundles."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return str(Path(meipass) / "assets" / "icons" / name)
    # thumbnail_grid.py lives at src/photoaident/ui/widgets/ — go up 5 levels
    return str(
        Path(__file__).parent.parent.parent.parent.parent / "assets" / "icons" / name
    )


def _reveal_in_file_manager(file_path: str) -> None:
    """Open the parent folder of file_path in the system file manager."""
    p = Path(file_path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(p)])
    else:  # Linux / BSD
        # Use D-Bus org.freedesktop.FileManager1 — the Linux equivalent of
        # Android intents. We call the already-running file manager directly
        # over D-Bus without spawning a subprocess, so the AppImage's
        # LD_LIBRARY_PATH never leaks into the file manager process.
        from PySide6 import QtDBus  # Linux-only module, import lazily

        file_uri = QtCore.QUrl.fromLocalFile(str(p)).toString()
        iface = QtDBus.QDBusInterface(
            "org.freedesktop.FileManager1",
            "/org/freedesktop/FileManager1",
            "org.freedesktop.FileManager1",
        )
        if iface.isValid():
            reply = iface.call("ShowItems", [file_uri], "")
            if reply.type() != QtDBus.QDBusMessage.MessageType.ErrorMessage:
                return
        # Fallback: no D-Bus file manager service, or the call was rejected.
        # Strip the AppImage library path so xdg-open's target process
        # won't pick up the bundled Qt libs.
        env = os.environ.copy()
        orig = env.pop("LD_LIBRARY_PATH_ORIG", None)
        if orig is not None:
            env["LD_LIBRARY_PATH"] = orig
        else:
            env.pop("LD_LIBRARY_PATH", None)
        subprocess.Popen(["xdg-open", str(p.parent)], env=env)


def _has_unidentified_faces(session_factory: "sessionmaker", image_id: int) -> bool:
    """Return True if the image has at least one unidentified, non-deleted face."""
    with session_factory() as session:
        n = session.scalar(
            select(func.count(Face.id)).where(
                Face.image_id == image_id,
                Face.state == FaceState.UNIDENTIFIED,
                Face.deleted_at.is_(None),
            )
        )
        return (n or 0) > 0


def _get_scaled_size(path: Path) -> QtCore.QSize:
    """Return the size to render 'path' within a 150×150 box, keeping aspect ratio."""
    reader = QtGui.QImageReader(str(path))
    if not reader.canRead():
        return QtCore.QSize()
    orig_size = reader.size()
    if not orig_size.isValid():
        return QtCore.QSize()
    w = orig_size.width()
    h = orig_size.height()
    if w <= 0 or h <= 0:
        return QtCore.QSize()
    scale = min(_THUMBNAIL_MAX_SIZE / w, _THUMBNAIL_MAX_SIZE / h)
    return QtCore.QSize(int(w * scale), int(h * scale))


def _read_pixmap(target_path: Path, scaled_size: QtCore.QSize) -> QtGui.QPixmap:
    """Read and scale an image file into a QPixmap."""
    if not scaled_size.isValid():
        return QtGui.QPixmap()
    reader = QtGui.QImageReader(str(target_path))
    reader.setScaledSize(scaled_size)
    image = reader.read()
    if image.isNull():
        return QtGui.QPixmap()
    return QtGui.QPixmap.fromImage(image)


class _HoverOverlay(QtWidgets.QWidget):
    """Semi-transparent hover overlay with View, Browse, and Label buttons."""

    view_requested = QtCore.Signal()
    label_requested = QtCore.Signal()

    def __init__(
        self,
        image_id: int,
        file_path: str,
        session_factory: "sessionmaker | None",
        parent=None,
    ):
        super().__init__(parent)
        self._image_id = image_id
        self._file_path = file_path
        self._session_factory = session_factory

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(_THUMBNAIL_WIDGET_SIZE, _THUMBNAIL_WIDGET_SIZE)

        # WA_TranslucentBackground breaks palette-based button rendering, so we
        # use explicit rgba values.  autoRaise=False is required so the button
        # frame is drawn unconditionally (not only on native hover).
        self.setStyleSheet("""
            QToolButton {
                background-color: rgba(240, 240, 240, 255);
                border: 1px solid rgba(100, 100, 100, 255);
                border-radius: 6px;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 255);
                border-color: rgba(60, 120, 220, 220);
            }
            QToolButton:pressed {
                background-color: rgba(200, 215, 245, 255);
                border-color: rgba(60, 120, 220, 255);
            }
            QToolButton:disabled {
                background-color: rgba(200, 200, 200, 196);
                border-color: rgba(100, 100, 100, 196);
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # View button — left, fills available height
        self.view_btn = QtWidgets.QToolButton()
        self.view_btn.setAutoRaise(False)
        self.view_btn.setIcon(QtGui.QIcon(_icon_path("view.svg")))
        self.view_btn.setIconSize(QtCore.QSize(48, 48))
        self.view_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.view_btn.setToolTip(self.tr("View details"))
        layout.addWidget(self.view_btn)

        # Right column: Browse (top) + Label (bottom), each fixed
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.browse_btn = QtWidgets.QToolButton()
        self.browse_btn.setAutoRaise(False)
        self.browse_btn.setIcon(QtGui.QIcon(_icon_path("browse.svg")))
        self.browse_btn.setIconSize(QtCore.QSize(24, 24))
        self.browse_btn.setFixedSize(70, 70)
        self.browse_btn.setToolTip(self.tr("Show in file manager"))
        right_layout.addWidget(self.browse_btn)

        self.label_btn = QtWidgets.QToolButton()
        self.label_btn.setAutoRaise(False)
        self.label_btn.setIcon(QtGui.QIcon(_icon_path("label.svg")))
        self.label_btn.setIconSize(QtCore.QSize(24, 24))
        self.label_btn.setFixedSize(70, 70)
        self.label_btn.setToolTip(self.tr("Label faces"))
        right_layout.addWidget(self.label_btn)

        layout.addLayout(right_layout)

        self.view_btn.clicked.connect(self.view_requested.emit)
        self.browse_btn.clicked.connect(
            lambda: _reveal_in_file_manager(self._file_path)
        )
        self.label_btn.clicked.connect(self.label_requested.emit)

        self.hide()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 150))
        painter.end()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._session_factory is not None:
            has_faces = _has_unidentified_faces(self._session_factory, self._image_id)
            self.label_btn.setEnabled(has_faces)
        else:
            self.label_btn.setEnabled(False)


class ThumbnailWidget(QtWidgets.QWidget):
    """Displays a single image thumbnail."""

    clicked = QtCore.Signal(int)  # image_id
    label_clicked = QtCore.Signal(int)  # image_id

    def __init__(
        self,
        image_id: int,
        file_path: str,
        thumb_path: Path,
        session_factory: "sessionmaker | None" = None,
        parent=None,
    ):
        super().__init__(parent)
        self.image_id = image_id
        self.file_path = file_path
        self.thumb_path = thumb_path

        self.setFixedSize(_THUMBNAIL_WIDGET_SIZE, _THUMBNAIL_WIDGET_SIZE)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self._load_thumbnail()

        self._overlay = _HoverOverlay(image_id, file_path, session_factory, self)
        self._overlay.view_requested.connect(lambda: self.clicked.emit(self.image_id))
        self._overlay.label_requested.connect(
            lambda: self.label_clicked.emit(self.image_id)
        )

    def _load_thumbnail(self):
        if not self.thumb_path.exists():
            try:
                generate_thumbnail(Path(self.file_path), self.thumb_path)
            except Exception as e:
                logger.warning(
                    "Error generating thumbnail for %s: %s", self.file_path, e
                )

        target_path = (
            self.thumb_path if self.thumb_path.exists() else Path(self.file_path)
        )
        scaled_size = _get_scaled_size(target_path)
        pixmap = _read_pixmap(target_path, scaled_size)

        if pixmap.isNull():
            self.image_label.setText(self.tr("Error loading image"))
            return

        self.image_label.setPixmap(pixmap)

    def event(self, ev: QtCore.QEvent) -> bool:
        if ev.type() == QtCore.QEvent.Type.HoverEnter:
            self._overlay.show()
        elif ev.type() == QtCore.QEvent.Type.HoverLeave:
            self._overlay.hide()
        return super().event(ev)


class ThumbnailGrid(QtWidgets.QWidget):
    """A scrollable grid of thumbnails with infinite scroll."""

    image_selected = QtCore.Signal(int)
    navigate_to_labelling = QtCore.Signal(int)
    results_changed = QtCore.Signal(int)  # total result count, emitted on set_results()
    page_loaded = QtCore.Signal(int, int)  # (loaded_so_far, total)

    def __init__(self, session_factory: "sessionmaker | None" = None, parent=None):
        super().__init__(parent)
        self._session_factory = session_factory

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        self.container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.container)
        self.grid_layout.setSpacing(_THUMBNAIL_GRID_SPACING)
        self.grid_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )

        self.scroll_area.setWidget(self.container)

        self._hint_label = QtWidgets.QLabel()
        self._hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._hint_label.hide()
        self.main_layout.addWidget(self._hint_label)

        self.thumbnails: list = []
        self.cols = 4

        self._all_results: list[tuple[int, str, Path]] = []
        self._loaded_count: int = 0

        self.scroll_area.verticalScrollBar().valueChanged.connect(
            self._on_scroll_changed
        )

        self.image_selected.connect(self._on_image_selected)

    def _on_image_selected(self, image_id: int) -> None:
        if self._session_factory is None:
            return
        with self._session_factory() as session:
            stmt = (
                select(Image)
                .where(Image.id == image_id)
                .options(joinedload(Image.faces), joinedload(Image.metadata_rel))
            )
            image = session.execute(stmt).unique().scalar_one_or_none()
            if image:
                dialog = ImageDetailDialog(image, self)
                dialog.navigate_to_labelling.connect(self.navigate_to_labelling.emit)
                dialog.exec()

    def clear(self):
        self._hint_label.hide()
        self._all_results = []
        self._loaded_count = 0

        # Clear thumbnails list before deleting widgets to avoid stale references
        self.thumbnails = []

        # Remove all widgets from the grid layout properly
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

    def add_thumbnail(
        self,
        image_id: int,
        file_path: str,
        thumb_path: Path,
    ):
        thumb = ThumbnailWidget(image_id, file_path, thumb_path, self._session_factory)
        thumb.clicked.connect(self.image_selected.emit)
        thumb.label_clicked.connect(self.navigate_to_labelling.emit)

        idx = len(self.thumbnails)
        row = idx // self.cols
        col = idx % self.cols

        self.grid_layout.addWidget(thumb, row, col)
        self.thumbnails.append(thumb)

    def set_results(self, results: Iterable[tuple[int, str, Path]]) -> None:
        self.clear()
        self._all_results = list(results)
        self._load_next_page()
        self.results_changed.emit(len(self._all_results))

    def _load_next_page(self):
        batch = self._all_results[self._loaded_count : self._loaded_count + PAGE_SIZE]
        for image_id, file_path, thumb_path in batch:
            self.add_thumbnail(image_id, file_path, thumb_path)
        self._loaded_count += len(batch)
        self._update_scroll_hint()
        self.page_loaded.emit(self._loaded_count, len(self._all_results))
        QtCore.QTimer.singleShot(0, self._check_fill_viewport)

    def _check_fill_viewport(self):
        if not shiboken6.isValid(self):
            return
        if (
            self.scroll_area.verticalScrollBar().maximum() == 0
            and self._loaded_count < len(self._all_results)
        ):
            self._load_next_page()

    def _update_scroll_hint(self):
        remaining = len(self._all_results) - self._loaded_count
        if remaining > 0:
            self._hint_label.setText(
                self.tr("Scroll to load more\u2026 ({n} remaining)").format(n=remaining)
            )
            self._hint_label.show()
        else:
            self._hint_label.hide()

    def _on_scroll_changed(self, value: int):
        scrollbar = self.scroll_area.verticalScrollBar()
        if (
            value >= scrollbar.maximum() - _SCROLL_LOAD_MARGIN
            and self._loaded_count < len(self._all_results)
        ):
            self._load_next_page()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Recalculate columns based on width
        # Subtract some margin for scrollbar
        new_cols = max(
            1,
            (self.width() - _SCROLL_AREA_MARGIN)
            // (_THUMBNAIL_WIDGET_SIZE + _THUMBNAIL_GRID_SPACING),
        )
        if new_cols != self.cols:
            self.cols = new_cols
            self._rearrange_grid()

    def _rearrange_grid(self):
        for i, thumb in enumerate(self.thumbnails):
            self.grid_layout.removeWidget(thumb)
            row = i // self.cols
            col = i % self.cols
            self.grid_layout.addWidget(thumb, row, col)
