from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select

from photoaident.db.database import Face, FaceState, Image as DBImage, Person
from photoaident.utils.file_manager import reveal_in_file_manager

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore


def _resolve_best_person_name(
    faiss_id: int,
    session_factory: "sessionmaker",
    vector_store: "VectorStore",
    threshold: float = 0.35,
) -> tuple[str, float] | None:
    """Find the best-matching identified person for an unidentified face embedding.

    Args:
        faiss_id: The FAISS index ID of the face to resolve.
        session_factory: SQLAlchemy session factory.
        vector_store: FAISS vector store.
        threshold: Minimum similarity score to consider a match.

    Returns:
        A (person_name, score) tuple, or None if no match found.
    """
    try:
        embedding = vector_store.get_embedding(faiss_id)
    except IndexError:
        return None

    # Search for top-11 similar faces so we can exclude self
    neighbors = vector_store.search(embedding, k=11, threshold=threshold)
    neighbor_ids = [nid for nid, _ in neighbors if nid != faiss_id]
    if not neighbor_ids:
        return None

    with session_factory() as session:
        stmt = (
            select(Face.faiss_id, Person.name)
            .join(Face.person)
            .where(
                Face.faiss_id.in_(neighbor_ids),
                Face.state == FaceState.IDENTIFIED,
                Face.deleted_at.is_(None),
            )
        )
        rows = session.execute(stmt).all()

    if not rows:
        return None

    id_to_score = {nid: score for nid, score in neighbors if nid != faiss_id}
    id_to_name = {row.faiss_id: row.name for row in rows}

    best_id = max(id_to_name.keys(), key=lambda nid: id_to_score.get(nid, 0.0))
    return id_to_name[best_id], id_to_score[best_id]


class _FaceOverlayLabel(QtWidgets.QLabel):
    """QLabel that shows a tooltip when the mouse hovers over a face bounding box."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._face_regions: list[tuple[QtCore.QRectF, str]] = []
        self._original_size: QtCore.QSize = QtCore.QSize()
        self.setMouseTracking(True)

    def set_face_regions(
        self,
        regions: list[tuple[QtCore.QRectF, str]],
        original_size: QtCore.QSize,
    ) -> None:
        """Set face bounding boxes (in original-image coords) and their tooltip text."""
        self._face_regions = regions
        self._original_size = original_size

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if not self._face_regions or not self._original_size.isValid():
            return

        pm = self.pixmap()
        if pm is None or pm.isNull():
            return

        pm_w = pm.width()
        pm_h = pm.height()
        if pm_w == 0 or pm_h == 0:
            return

        # Center-alignment offsets (label may be larger than the pixmap)
        offset_x = (self.width() - pm_w) / 2
        offset_y = (self.height() - pm_h) / 2

        pos = event.position()
        orig_x = (pos.x() - offset_x) / pm_w * self._original_size.width()
        orig_y = (pos.y() - offset_y) / pm_h * self._original_size.height()

        for rect, tooltip_text in self._face_regions:
            if rect.contains(orig_x, orig_y):
                QtWidgets.QToolTip.showText(
                    event.globalPosition().toPoint(), tooltip_text, self
                )
                return

        QtWidgets.QToolTip.hideText()


class ImageDetailDialog(QtWidgets.QDialog):
    """
    A modal dialog that displays a full-size image with its metadata and
    face bounding boxes.
    """

    navigate_to_labelling = QtCore.Signal(int)  # image_id

    def __init__(
        self,
        image: DBImage,
        session_factory: "sessionmaker | None" = None,
        vector_store: "VectorStore | None" = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Details"))
        self.setMinimumSize(800, 600)
        self.image_data = image
        self._session_factory = session_factory
        self._vector_store = vector_store

        self._setup_ui()
        self._load_image()

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left panel: Metadata
        metadata_panel = QtWidgets.QFrame()
        metadata_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        metadata_panel.setFixedWidth(250)
        metadata_layout = QtWidgets.QVBoxLayout(metadata_panel)
        metadata_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        title_label = QtWidgets.QLabel("<b>" + self.tr("Metadata") + "</b>")
        metadata_layout.addWidget(title_label)

        # Helper to add metadata rows
        def add_meta(label_text, value_text):
            row_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"<b>{label_text}:</b>")
            label.setFixedWidth(80)
            value = QtWidgets.QLabel(str(value_text))
            value.setWordWrap(True)
            value.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            row_layout.addWidget(label)
            row_layout.addWidget(value)
            metadata_layout.addLayout(row_layout)

        add_meta(self.tr("ID"), self.image_data.id)
        add_meta(self.tr("File Path"), self.image_data.file_path)

        # Format file size
        size_bytes = self.image_data.file_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        add_meta(self.tr("File Size"), size_str)

        if self.image_data.metadata_rel:
            meta = self.image_data.metadata_rel
            if meta.width and meta.height:
                add_meta(self.tr("Dimensions"), f"{meta.width} x {meta.height}")
            if meta.taken_at:
                add_meta(
                    self.tr("Taken At"), meta.taken_at.strftime("%Y-%m-%d %H:%M:%S")
                )
            if meta.camera_make or meta.camera_model:
                camera = f"{meta.camera_make or ''} {meta.camera_model or ''}".strip()
                add_meta(self.tr("Camera"), camera)

        metadata_layout.addStretch()

        has_unidentified = any(
            f.state == FaceState.UNIDENTIFIED and f.deleted_at is None
            for f in self.image_data.faces
        )
        label_btn = QtWidgets.QPushButton(self.tr("Label Faces"))
        label_btn.setEnabled(has_unidentified)
        label_btn.clicked.connect(self._on_label_faces_clicked)
        metadata_layout.addWidget(label_btn)

        show_in_file_manager_btn = QtWidgets.QPushButton(
            self.tr("Show in File Manager")
        )
        show_in_file_manager_btn.clicked.connect(self._on_show_in_file_manager_clicked)
        metadata_layout.addWidget(show_in_file_manager_btn)

        close_button = QtWidgets.QPushButton(self.tr("Close"))
        close_button.clicked.connect(self.accept)
        metadata_layout.addWidget(close_button)

        main_layout.addWidget(metadata_panel)

        # Right panel: Image View
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.image_label = _FaceOverlayLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        main_layout.addWidget(self.scroll_area, 1)

    def _build_face_display_info(
        self,
    ) -> list[tuple[QtCore.QRectF, QtCore.Qt.GlobalColor, str]]:
        """Return (rect, color, tooltip) for every non-deleted face.

        Color is green for identified/anonymous/matched-unidentified faces and
        red only for truly unknown (unidentified with no FAISS match) faces.
        """
        _GREEN = QtCore.Qt.GlobalColor.green
        _RED = QtCore.Qt.GlobalColor.red

        result: list[tuple[QtCore.QRectF, QtCore.Qt.GlobalColor, str]] = []
        for face in self.image_data.faces:
            if face.deleted_at is not None:
                continue

            rect = QtCore.QRectF(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)

            if face.state == FaceState.IDENTIFIED:
                tooltip = (
                    face.person.name if face.person is not None else self.tr("Unknown")
                )
                color = _GREEN
            elif face.state == FaceState.ANONYMOUS:
                tooltip = self.tr("Anonymous")
                color = _GREEN
            else:  # UNIDENTIFIED
                if self._vector_store is not None and self._session_factory is not None:
                    match = _resolve_best_person_name(
                        face.faiss_id, self._session_factory, self._vector_store
                    )
                    if match is not None:
                        name, score = match
                        tooltip = f"{name} ({score:.0%})"
                        color = _GREEN
                    else:
                        tooltip = self.tr("Unknown")
                        color = _RED
                else:
                    tooltip = self.tr("Unknown")
                    color = _RED

            result.append((rect, color, tooltip))
        return result

    def _load_image(self):
        file_path = Path(self.image_data.file_path)
        if not file_path.exists():
            self.image_label.setText(
                self.tr("Image file not found: {path}").format(path=file_path)
            )
            return

        # Load image with QImageReader to respect EXIF orientation
        reader = QtGui.QImageReader(str(file_path))
        reader.setAutoTransform(True)
        qimage = reader.read()

        if qimage.isNull():
            self.image_label.setText(
                self.tr("Failed to load image: {error}").format(
                    error=reader.errorString()
                )
            )
            return

        pixmap = QtGui.QPixmap.fromImage(qimage)

        # Build display info first so colors reflect FAISS match results
        face_display = self._build_face_display_info()

        if face_display:
            pen_width = max(2, pixmap.width() // 500)
            painter = QtGui.QPainter(pixmap)
            for rect, color, _ in face_display:
                pen = QtGui.QPen(color)
                pen.setWidth(pen_width)
                painter.setPen(pen)
                painter.drawRect(rect.toRect())
            painter.end()

        # Register tooltip regions (rect + text only)
        tooltip_regions = [(rect, tooltip) for rect, _, tooltip in face_display]
        self.image_label.set_face_regions(tooltip_regions, pixmap.size())

        self._original_pixmap = pixmap
        self._update_image_display()

    def _update_image_display(self):
        if not hasattr(self, "_original_pixmap"):
            return

        # Scaling logic to fit nicely in the dialog but keep it readable
        available_size = self.scroll_area.viewport().size()
        scaled_pixmap = self._original_pixmap.scaled(
            available_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def _on_label_faces_clicked(self) -> None:
        self.accept()
        self.navigate_to_labelling.emit(self.image_data.id)

    def _on_show_in_file_manager_clicked(self) -> None:
        reveal_in_file_manager(self.image_data.file_path)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Re-scale image when dialog is resized
        QtCore.QTimer.singleShot(10, self._update_image_display)
