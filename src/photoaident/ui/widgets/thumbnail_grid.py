from pathlib import Path
from typing import List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.utils.image_utils import generate_thumbnail


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
    scale = min(150 / w, 150 / h)
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


class ThumbnailWidget(QtWidgets.QWidget):
    """Displays a single image thumbnail."""

    clicked = QtCore.Signal(int)  # image_id

    def __init__(
        self,
        image_id: int,
        file_path: str,
        thumb_path: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.image_id = image_id
        self.file_path = file_path
        self.thumb_path = thumb_path

        self.setFixedSize(160, 160)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self._load_thumbnail()

    def _load_thumbnail(self):
        if not self.thumb_path.exists():
            try:
                generate_thumbnail(Path(self.file_path), self.thumb_path)
            except Exception as e:
                print(f"Error generating thumbnail for {self.file_path}: {e}")

        target_path = (
            self.thumb_path if self.thumb_path.exists() else Path(self.file_path)
        )
        scaled_size = _get_scaled_size(target_path)
        pixmap = _read_pixmap(target_path, scaled_size)

        if pixmap.isNull():
            self.image_label.setText(self.tr("Error loading image"))
            return

        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_id)
        super().mousePressEvent(event)


class ThumbnailGrid(QtWidgets.QWidget):
    """A scrollable grid of thumbnails."""

    image_selected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        self.container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.container)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )

        self.scroll_area.setWidget(self.container)

        self.thumbnails = []
        self.cols = 4

    def clear(self):
        # Remove the "Showing X of Y" label if it exists
        while self.main_layout.count() > 1:
            item = self.main_layout.itemAt(1)
            if (
                item
                and item.widget()
                and not isinstance(item.widget(), QtWidgets.QScrollArea)
            ):
                widget = item.widget()
                if widget:
                    self.main_layout.removeWidget(widget)
                    widget.deleteLater()
            else:
                # Should not happen as scroll area is at 0, but safety first
                break

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
        thumb = ThumbnailWidget(image_id, file_path, thumb_path)
        thumb.clicked.connect(self.image_selected.emit)

        idx = len(self.thumbnails)
        row = idx // self.cols
        col = idx % self.cols

        self.grid_layout.addWidget(thumb, row, col)
        self.thumbnails.append(thumb)

    def set_images_with_total(
        self,
        images_data: List[Tuple[int, str, Path]],
        total_count: int,
    ):
        self.clear()
        if not images_data:
            # If no images, still show total if it's > 0 (though this shouldn't happen)
            if total_count > 0:
                label = QtWidgets.QLabel(self.tr("No images found."))
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.main_layout.insertWidget(1, label)
            return

        for img_id, path, thumb_path in images_data:
            self.add_thumbnail(img_id, path, thumb_path)

        if total_count > len(images_data):
            label = QtWidgets.QLabel(
                self.tr("Showing first {limit} of {total} images.").format(
                    limit=len(images_data), total=total_count
                )
            )
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.main_layout.insertWidget(1, label)

    def set_images(
        self,
        images_data: List[Tuple[int, str, Path]],
    ):
        self.set_images_with_total(images_data, len(images_data))

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Recalculate columns based on width
        # Subtract some margin for scrollbar
        new_cols = max(1, (self.width() - 20) // 170)
        if new_cols != self.cols:
            self.cols = new_cols
            self._rearrange_grid()

    def _rearrange_grid(self):
        for i, thumb in enumerate(self.thumbnails):
            self.grid_layout.removeWidget(thumb)
            row = i // self.cols
            col = i % self.cols
            self.grid_layout.addWidget(thumb, row, col)
