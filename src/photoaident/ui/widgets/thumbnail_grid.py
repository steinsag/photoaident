from pathlib import Path
from typing import List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from photoaident.db.database import Face


class ThumbnailWidget(QtWidgets.QWidget):
    """Displays a single image thumbnail with face highlights."""

    clicked = QtCore.Signal(int)  # image_id

    def __init__(
        self,
        image_id: int,
        file_path: str,
        faces: List[Face],
        thumb_path: Path,
        orig_size: Tuple[int, int] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.image_id = image_id
        self.file_path = file_path
        self.faces = faces
        self.thumb_path = thumb_path
        self.orig_size = orig_size

        self.setFixedSize(160, 160)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        self._load_thumbnail()

    def _load_thumbnail(self):
        # Calculate scaled size while keeping aspect ratio
        def get_scaled_size(path):
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

        target_path = (
            self.thumb_path if self.thumb_path.exists() else Path(self.file_path)
        )
        scaled_size = get_scaled_size(target_path)

        if scaled_size.isValid():
            reader = QtGui.QImageReader(str(target_path))
            reader.setScaledSize(scaled_size)
            image = reader.read()
            if image.isNull():
                pixmap = QtGui.QPixmap()
            else:
                pixmap = QtGui.QPixmap.fromImage(image)
        else:
            pixmap = QtGui.QPixmap()

        if pixmap.isNull():
            self.image_label.setText(self.tr("Error loading image"))
            return

        # Draw faces
        if self.faces and self.orig_size:
            orig_w, orig_h = self.orig_size
            if orig_w > 0 and orig_h > 0:
                painter = QtGui.QPainter(pixmap)
                pen = QtGui.QPen(QtGui.QColor("red"), 2)
                painter.setPen(pen)

                # pixmap size
                pw = pixmap.width()
                ph = pixmap.height()

                # Scale factor (pixmap is KeepAspectRatio)
                scale = min(pw / orig_w, ph / orig_h)

                # Offsets if pixmap is centered (it might not be if scaled to fit
                # exactly). But here pixmap is the result of scaled(), so it
                # has exactly the scaled size.

                for face in self.faces:
                    # Face bboxes are [x, y, w, h] or [x1, y1, x2, y2]?
                    # Looking at database.py: bbox_x, bbox_y, bbox_w, bbox_h
                    fx = face.bbox_x * scale
                    fy = face.bbox_y * scale
                    fw = face.bbox_w * scale
                    fh = face.bbox_h * scale
                    painter.drawRect(QtCore.QRectF(fx, fy, fw, fh))

                painter.end()

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
        if self.main_layout.count() > 1:
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
        faces: List[Face],
        thumb_path: Path,
        orig_size: Tuple[int, int] | None = None,
    ):
        thumb = ThumbnailWidget(image_id, file_path, faces, thumb_path, orig_size)
        thumb.clicked.connect(self.image_selected.emit)

        idx = len(self.thumbnails)
        row = idx // self.cols
        col = idx % self.cols

        self.grid_layout.addWidget(thumb, row, col)
        self.thumbnails.append(thumb)

    def set_images_with_total(
        self,
        images_data: List[Tuple[int, str, List[Face], Path, Tuple[int, int] | None]],
        total_count: int,
    ):
        self.clear()
        if not images_data:
            return

        for img_id, path, faces, thumb_path, orig_size in images_data:
            self.add_thumbnail(img_id, path, faces, thumb_path, orig_size)

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
        images_data: List[Tuple[int, str, List[Face], Path, Tuple[int, int] | None]],
    ):
        self.set_images_with_total(images_data, len(images_data))

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Recalculate columns based on width
        new_cols = max(1, self.width() // 170)
        if new_cols != self.cols:
            self.cols = new_cols
            self._rearrange_grid()

    def _rearrange_grid(self):
        for i, thumb in enumerate(self.thumbnails):
            self.grid_layout.removeWidget(thumb)
            row = i // self.cols
            col = i % self.cols
            self.grid_layout.addWidget(thumb, row, col)
