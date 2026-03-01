from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


def _exif_pixmap_transform(
    transformation: QtGui.QImageIOHandler.Transformation,
) -> QtGui.QTransform:
    """Return the QTransform that corresponds to a QImageIOHandler.Transformation.

    Replicates the logic Qt applies internally when
    ``QImageReader.setAutoTransform(True)`` is used, so callers can apply the
    same orientation correction manually after drawing overlays in the
    un-rotated coordinate space.

    The implementation mirrors the switch statement in Qt's own
    ``exifTransform`` helper (qtbase/src/gui/image/qimagereader.cpp).
    """
    m = QtGui.QTransform()
    T = QtGui.QImageIOHandler.Transformation
    if transformation == T.TransformationMirror:
        m.scale(-1.0, 1.0)
    elif transformation == T.TransformationFlip:
        m.scale(1.0, -1.0)
    elif transformation == T.TransformationRotate180:
        m.rotate(180.0)
    elif transformation == T.TransformationRotate90:
        m.rotate(90.0)
    elif transformation == T.TransformationMirrorAndRotate90:
        m.scale(-1.0, 1.0)
        m.rotate(90.0)
    elif transformation == T.TransformationFlipAndRotate90:
        m.scale(1.0, -1.0)
        m.rotate(90.0)
    elif transformation == T.TransformationRotate270:
        m.rotate(270.0)
    # TransformationNone → identity, already initialised above
    return m


class ImagePreviewWidget(QtWidgets.QWidget):
    """Displays a full photo with a highlighted face bounding box."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._original_pixmap: Optional[QtGui.QPixmap] = None
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

        reader = QtGui.QImageReader(str(image_path))
        # Do NOT call setAutoTransform: InsightFace/OpenCV bbox coordinates are
        # in the un-rotated pixel space (cv2.imread ignores EXIF orientation).
        # We draw the bbox in that space first, then rotate the whole pixmap so
        # the box stays correctly aligned with the face after orientation is
        # applied.
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

        exif_transform = _exif_pixmap_transform(reader.transformation())
        if not exif_transform.isIdentity():
            pixmap = pixmap.transformed(
                exif_transform, QtCore.Qt.TransformationMode.SmoothTransformation
            )

        self._original_pixmap = pixmap
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
