from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select

from photoaident.db.database import Face, FaceState, Image as DBImage, Person
from photoaident.paths import AppPaths
from photoaident.ui.window_state import restore_widget_geometry, save_widget_geometry
from photoaident.utils.file_manager import reveal_in_file_manager
from photoaident.utils.image_utils import get_exif_transform

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from photoaident.db.vector_store import VectorStore


def _resolve_batch_person_names(
    faiss_ids: list[int],
    session: "Session",
    vector_store: "VectorStore",
    threshold: float = 0.35,
) -> dict[int, tuple[str, float] | None]:
    """Find the best-matching identified person for several unidentified
    face embeddings.

    Args:
        faiss_ids: List of FAISS index IDs of the faces to resolve.
        session: SQLAlchemy session (already open).
        vector_store: FAISS vector store.
        threshold: Minimum similarity score to consider a match.

    Returns:
        A dictionary mapping each faiss_id to a (person_name, score) tuple or None.
    """
    if not faiss_ids:
        return {}

    # 1. Collect all neighbor searches
    all_neighbors_map: dict[int, list[tuple[int, float]]] = {}
    all_neighbor_ids: set[int] = set()

    for fid in faiss_ids:
        try:
            embedding = vector_store.get_embedding(fid)
            # Search for top-11 similar faces so we can exclude self
            neighbors = vector_store.search(embedding, k=11, threshold=threshold)
            neighbor_ids = [nid for nid, _ in neighbors if nid != fid]
            if neighbor_ids:
                all_neighbors_map[fid] = neighbors
                all_neighbor_ids.update(neighbor_ids)
        except IndexError:
            continue

    if not all_neighbor_ids:
        return dict.fromkeys(faiss_ids)

    # 2. Batch DB query for all potential neighbor names
    stmt = (
        select(Face.faiss_id, Person.name)
        .join(Face.person)
        .where(
            Face.faiss_id.in_(list(all_neighbor_ids)),
            Face.state == FaceState.IDENTIFIED,
            Face.deleted_at.is_(None),
        )
    )
    rows = session.execute(stmt).all()
    id_to_name = {row.faiss_id: row.name for row in rows}

    # 3. Map back to each original face
    results: dict[int, tuple[str, float] | None] = {}
    for fid in faiss_ids:
        neighbors = all_neighbors_map.get(fid)
        if not neighbors:
            results[fid] = None
            continue

        # Filter neighbors to those that have an identified name in our DB result
        valid_neighbors = [
            (nid, score) for nid, score in neighbors if nid in id_to_name and nid != fid
        ]

        if not valid_neighbors:
            results[fid] = None
            continue

        # Pick the one with the highest score
        best_id, best_score = max(valid_neighbors, key=lambda x: x[1])
        results[fid] = (id_to_name[best_id], best_score)

    return results


def _resolve_best_person_name(
    faiss_id: int,
    session: "Session",
    vector_store: "VectorStore",
    threshold: float = 0.35,
) -> tuple[str, float] | None:
    """Find the best-matching identified person for an unidentified face embedding.

    Args:
        faiss_id: The FAISS index ID of the face to resolve.
        session: SQLAlchemy session (already open).
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

    return _query_best_person(faiss_id, neighbor_ids, neighbors, session)


def _query_best_person(
    faiss_id: int,
    neighbor_ids: list[int],
    neighbors: list[tuple[int, float]],
    session: "Session",
) -> tuple[str, float] | None:
    """Execute the DB query to find the best person among neighbors."""
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
        self._last_hovered_index: int = -1
        self.setMouseTracking(True)

    def set_face_regions(
        self,
        pixmap_regions: list[tuple[QtCore.QRectF, str]],
        pixmap_size: QtCore.QSize,
    ) -> None:
        """
        Set face bounding boxes and their tooltip text.
        Coordinates must be in the same coordinate space as the pixmap
        provided to setPixmap (e.g., after EXIF transformation but before
        UI-level scaling).
        """
        self._face_regions = pixmap_regions
        self._original_size = pixmap_size
        self._last_hovered_index = -2

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
        # Map mouse position to pixmap coordinates
        px_x = (pos.x() - offset_x) / pm_w * self._original_size.width()
        px_y = (pos.y() - offset_y) / pm_h * self._original_size.height()
        current_hovered_index = -1

        for i, (rect, tooltip_text) in enumerate(self._face_regions):
            if rect.contains(px_x, px_y):
                current_hovered_index = i
                break

        if current_hovered_index != self._last_hovered_index:
            self._last_hovered_index = current_hovered_index
            if current_hovered_index != -1:
                _, tooltip_text = self._face_regions[current_hovered_index]
                QtWidgets.QToolTip.showText(
                    event.globalPosition().toPoint(), tooltip_text, self
                )
            else:
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
        session_factory: "sessionmaker",
        vector_store: "VectorStore",
        paths: "AppPaths",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Details"))
        self.setMinimumSize(800, 600)
        self.image_data = image
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._paths = paths
        self._resolved_names: dict[int, tuple[str, float] | None] = {}

        self._setup_ui()
        self._load_image()
        restore_widget_geometry(self, self._paths.window_state_file)

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in bytes to a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _setup_ui(self) -> None:
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
        def add_meta(label_text: str, value_text: str | int | None) -> None:
            if value_text is None:
                return
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
        add_meta(
            self.tr("File Size"), self._format_file_size(self.image_data.file_size)
        )

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

    def _get_face_display_info(self, face: Face) -> tuple[QtCore.Qt.GlobalColor, str]:
        """Return (color, tooltip) for a single face based on its state."""
        green = QtCore.Qt.GlobalColor.green
        red = QtCore.Qt.GlobalColor.red

        if face.state == FaceState.IDENTIFIED and face.person:
            return green, face.person.name

        if face.state == FaceState.ANONYMOUS:
            return green, self.tr("Anonymous")

        # UNIDENTIFIED (or IDENTIFIED without a person): check cached FAISS match
        match = self._resolved_names.get(face.faiss_id)
        if match:
            name, score = match
            return green, f"{name} ({score:.0%})"

        return red, self.tr("Unknown")

    def _build_face_display_info(
        self,
    ) -> list[tuple[QtCore.QRectF, QtCore.Qt.GlobalColor, str]]:
        """Return (rect, color, tooltip) for every non-deleted face.

        Color is green for identified/anonymous/matched-unidentified faces and
        red only for truly unknown (unidentified with no FAISS match) faces.
        """
        # Pre-resolve unidentified faces in a single session to avoid N+1 queries
        unidentified_ids = [
            f.faiss_id
            for f in self.image_data.faces
            if f.state == FaceState.UNIDENTIFIED
            and f.deleted_at is None
            and f.faiss_id not in self._resolved_names
        ]

        if unidentified_ids:
            with self._session_factory() as session:
                resolved = _resolve_batch_person_names(
                    unidentified_ids, session, self._vector_store
                )
                self._resolved_names.update(resolved)

        result: list[tuple[QtCore.QRectF, QtCore.Qt.GlobalColor, str]] = []
        for face in self.image_data.faces:
            if face.deleted_at is not None:
                continue

            rect = QtCore.QRectF(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h)
            color, tooltip = self._get_face_display_info(face)
            result.append((rect, color, tooltip))

        return result

    def _load_image(self):
        file_path = Path(self.image_data.file_path)
        if not file_path.exists():
            self.image_label.setText(
                self.tr("Image file not found: {path}").format(path=file_path)
            )
            return

        # Load image with QImageReader. We do NOT call setAutoTransform(True) here
        # because InsightFace/OpenCV bbox coordinates are in the un-rotated
        # pixel space. We draw the bbox in that space first, then rotate the
        # whole pixmap so the box stays correctly aligned with the face.
        reader = QtGui.QImageReader(str(file_path))
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

        # Apply EXIF transformation after drawing bounding boxes
        exif_transform = get_exif_transform(reader.transformation())
        if not exif_transform.isIdentity():
            # Get the matrix that includes necessary translations
            # to stay within the bounds of the transformed pixmap.
            true_transform = QtGui.QPixmap.trueMatrix(
                exif_transform, pixmap.width(), pixmap.height()
            )
            pixmap = pixmap.transformed(
                exif_transform, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            # Use true_transform for mapping rects so they land in [0, new_width/height]
            exif_transform = true_transform

        # Register tooltip regions (rect + text only)
        # Note: Bounding boxes are in un-rotated space, but tooltip hit-testing
        # needs to account for the rotation. We transform the rects as well.
        tooltip_regions = []
        for rect, _, tooltip in face_display:
            transformed_rect = exif_transform.mapRect(rect)
            tooltip_regions.append((transformed_rect, tooltip))

        self.image_label.set_face_regions(
            pixmap_regions=tooltip_regions, pixmap_size=pixmap.size()
        )

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

    def done(self, result: int) -> None:
        """Save geometry before closing."""
        save_widget_geometry(self, self._paths.window_state_file)
        super().done(result)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        # Re-scale image when dialog is resized
        QtCore.QTimer.singleShot(10, self._update_image_display)
