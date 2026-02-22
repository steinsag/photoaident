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
        if self.thumb_path.exists():
            pixmap = QtGui.QPixmap(str(self.thumb_path))
        else:
            # Fallback to original file if thumbnail doesn't exist yet
            # In a real app, we'd trigger thumbnail generation
            pixmap = QtGui.QPixmap(self.file_path)

        if pixmap.isNull():
            self.image_label.setText(self.tr("Error loading image"))
            return

        # Scale pixmap to fit
        pixmap = pixmap.scaled(
            150,
            150,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

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
                    widget.setParent(None)
                    widget.deleteLater()

        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    # Potential cause of segfault if called here:
                    # widget.setParent(None)
                    widget.deleteLater()
        self.thumbnails = []

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

    def set_images(
        self,
        images_data: List[Tuple[int, str, List[Face], Path, Tuple[int, int] | None]],
    ):
        self.clear()
        if not images_data:
            return

        # Cap the number of thumbnails to avoid performance issues/segfaults
        # with very large collections for now.
        # 1000 is a reasonable limit for a single page without virtualization.
        MAX_THUMBS = 1000
        for img_id, path, faces, thumb_path, orig_size in images_data[:MAX_THUMBS]:
            self.add_thumbnail(img_id, path, faces, thumb_path, orig_size)

        if len(images_data) > MAX_THUMBS:
            label = QtWidgets.QLabel(
                self.tr("Showing first {limit} of {total} images.").format(
                    limit=MAX_THUMBS, total=len(images_data)
                )
            )
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.main_layout.insertWidget(1, label)

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
