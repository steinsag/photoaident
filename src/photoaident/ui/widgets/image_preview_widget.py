from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


class ImagePreviewWidget(QtWidgets.QWidget):
    """Displays a full photo with a highlighted face bounding box."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._original_pixmap: QtGui.QPixmap | None = None
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(10)
        self._resize_timer.timeout.connect(self._update_display)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel()
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background-color: #222;")
        layout.addWidget(self._label)

    def load(self, image_path: Path, bbox: tuple[int, int, int, int]) -> None:
        """Load image from path and draw a red bounding box at bbox (x, y, w, h)."""
        if not image_path.exists():
            self._label.setPixmap(QtGui.QPixmap())
            self._label.setText(str(image_path))
            self._original_pixmap = None
            return

        # Load with EXIF orientation applied. Bounding boxes stored in the DB
        # are also in this EXIF-corrected coordinate space (process_image uses
        # open_image which calls ImageOps.exif_transpose before detection).
        reader = QtGui.QImageReader(str(image_path))
        reader.setAutoTransform(True)
        qimage = reader.read()
        if qimage.isNull():
            self._label.setPixmap(QtGui.QPixmap())
            self._label.setText(reader.errorString())
            self._original_pixmap = None
            return

        pixmap = QtGui.QPixmap.fromImage(qimage)

        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.red)
        pen.setWidth(max(2, pixmap.width() // 500))
        painter.setPen(pen)
        x, y, w, h = bbox
        painter.drawRect(QtCore.QRect(x, y, w, h))
        painter.end()

        self._original_pixmap = pixmap
        self._label.setText("")
        self._update_display()

    def clear(self) -> None:
        """Reset to empty state."""
        self._original_pixmap = None
        self._label.setPixmap(QtGui.QPixmap())
        self._label.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._resize_timer.start()

    def _update_display(self) -> None:
        if self._original_pixmap is None:
            return
        available = self._label.size()
        if available.isEmpty():
            return
        scaled = self._original_pixmap.scaled(
            available,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
