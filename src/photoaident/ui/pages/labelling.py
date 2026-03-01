import bisect
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import func, select
from sqlalchemy.orm import contains_eager, selectinload

from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
)
from photoaident.ui.widgets.new_person_dialog import NewPersonDialog

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths

_COL_NAME = 0
_COL_SCORE = 1


@dataclass(frozen=True, slots=True)
class _FaceData:
    face_id: int
    crop_path: Optional[Path]
    image_path: Path
    bbox: tuple[int, int, int, int]
    faiss_id: int


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


class _ImagePreviewWidget(QtWidgets.QWidget):
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


class LabellingPage(QtWidgets.QWidget):
    """Page for labelling unidentified faces one by one."""

    def __init__(
        self,
        session_factory: "sessionmaker",
        paths: "AppPaths",
        vector_store: "VectorStore",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
        self.paths = paths
        self.vector_store = vector_store
        self._current_face_id: Optional[int] = None
        self._skipped: set[int] = set()
        self._skipped_images: set[int] = set()
        self._priority_image_id: Optional[int] = None
        self._query_embedding: Optional[np.ndarray] = None
        self._all_persons: list[Person] = []
        self._selected_person: Optional[Person] = None
        self._selected_cluster: Optional[EmbeddingCluster] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # --- Top row: image preview | face crop | action buttons ---
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setSpacing(8)

        self._image_preview = _ImagePreviewWidget()
        self._image_preview.setMinimumSize(300, 300)
        top_layout.addWidget(self._image_preview, stretch=1)

        self._crop_label = QtWidgets.QLabel()
        self._crop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._crop_label.setWordWrap(True)
        self._crop_label.setFixedSize(300, 300)
        self._crop_label.setStyleSheet(
            "background-color: #aaa; border: 1px solid #888;"
        )
        top_layout.addWidget(self._crop_label)

        btn_col = QtWidgets.QVBoxLayout()
        btn_col.setSpacing(6)
        btn_col.addStretch()

        self.skip_image_btn = QtWidgets.QPushButton(self.tr("Skip Image"))
        self.skip_image_btn.clicked.connect(self._skip_image)
        btn_col.addWidget(self.skip_image_btn)

        self.skip_btn = QtWidgets.QPushButton(self.tr("Skip Face"))
        self.skip_btn.clicked.connect(self._skip_face)
        btn_col.addWidget(self.skip_btn)

        self.anonymous_btn = QtWidgets.QPushButton(self.tr("Mark Anonymous"))
        self.anonymous_btn.clicked.connect(self._mark_anonymous)
        btn_col.addWidget(self.anonymous_btn)

        btn_col.addStretch()
        top_layout.addLayout(btn_col)

        root_layout.addLayout(top_layout)

        # --- Bottom: inline person selector ---
        person_group = QtWidgets.QGroupBox(self.tr("Select Person"))
        group_layout = QtWidgets.QHBoxLayout(person_group)
        group_layout.setSpacing(8)

        # Left panel: filter + list + new person button
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(250)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._search_edit = QtWidgets.QLineEdit()
        self._search_edit.setPlaceholderText(self.tr("Type to filter"))
        self._search_edit.textChanged.connect(self._filter_persons)
        left_layout.addWidget(self._search_edit)

        self._person_list = QtWidgets.QListWidget()
        self._person_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._person_list.currentItemChanged.connect(self._on_person_selected)
        left_layout.addWidget(self._person_list, stretch=1)

        self._new_person_btn = QtWidgets.QPushButton(self.tr("New Person\u2026"))
        self._new_person_btn.clicked.connect(self._create_new_person)
        left_layout.addWidget(self._new_person_btn)

        group_layout.addWidget(left_panel)

        # Right panel: cluster table + confirm/cancel
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._cluster_table = QtWidgets.QTableWidget(5, 2)
        self._cluster_table.setHorizontalHeaderLabels(
            [self.tr("Age Group"), self.tr("Similarity")]
        )
        self._cluster_table.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._cluster_table.horizontalHeader().setSectionResizeMode(
            _COL_SCORE, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self._cluster_table.setColumnWidth(_COL_SCORE, 80)
        self._cluster_table.verticalHeader().setVisible(False)
        self._cluster_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._cluster_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._cluster_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        age_display_names = [
            self.tr("Infant (0\u20133)"),
            self.tr("Youngster (4\u201312)"),
            self.tr("Teenager (13\u201319)"),
            self.tr("Adult (20\u201375)"),
            self.tr("Senior (75+)"),
        ]
        for row, display_name in enumerate(age_display_names):
            name_item = QtWidgets.QTableWidgetItem(display_name)
            self._cluster_table.setItem(row, _COL_NAME, name_item)
            score_item = QtWidgets.QTableWidgetItem("\u2014")
            score_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self._cluster_table.setItem(row, _COL_SCORE, score_item)

        self._cluster_table.itemSelectionChanged.connect(self._on_cluster_row_selected)
        right_layout.addWidget(self._cluster_table, stretch=1)

        confirm_row = QtWidgets.QHBoxLayout()
        confirm_row.addStretch()

        self._cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        confirm_row.addWidget(self._cancel_btn)

        self.confirm_btn = QtWidgets.QPushButton(self.tr("Confirm"))
        self.confirm_btn.clicked.connect(self._on_confirm)
        confirm_row.addWidget(self.confirm_btn)

        right_layout.addLayout(confirm_row)
        group_layout.addWidget(right_panel, stretch=1)

        root_layout.addWidget(person_group)

        self._set_buttons_enabled(False)
        self._update_confirm_button()

    def refresh(self, priority_image_id: Optional[int] = None) -> None:
        """Reload count + first face. Called when page becomes visible."""
        self._skipped.clear()
        self._skipped_images.clear()
        self._priority_image_id = priority_image_id
        self._load_next_face()

    # ------------------------------------------------------------------
    # Private: face loading pipeline
    # ------------------------------------------------------------------

    def _load_next_face(self) -> None:
        with self.session_factory() as session:
            self._maybe_clear_priority(session)
            face_data = self._extract_next_face_data(session)

        if face_data is None:
            self._current_face_id = None
            self._query_embedding = None
            self._show_empty_state()
            return

        self._current_face_id = face_data.face_id

        # Fetch embedding for similarity scoring
        self._query_embedding = None
        try:
            self._query_embedding = self.vector_store.get_embedding(face_data.faiss_id)
        except Exception:
            logger.warning(
                "Failed to retrieve embedding for face %s (faiss_id=%s)",
                face_data.face_id,
                face_data.faiss_id,
                exc_info=True,
            )

        self._image_preview.load(face_data.image_path, face_data.bbox)
        self._load_crop(face_data.crop_path)
        self._set_buttons_enabled(True)
        self._load_persons()

    def _load_crop(self, crop_path: Optional[Path]) -> None:
        if crop_path is not None and crop_path.exists():
            pix = QtGui.QPixmap(str(crop_path))
            if not pix.isNull():
                self._crop_label.setText("")
                self._crop_label.setPixmap(
                    pix.scaledToHeight(
                        300, QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                )
                return
        self._crop_label.setPixmap(QtGui.QPixmap())
        self._crop_label.setText(self.tr("No image"))

    def _maybe_clear_priority(self, session: "Session") -> None:
        if self._priority_image_id is None:
            return
        if self._priority_image_id in self._skipped_images:
            self._priority_image_id = None
            return
        q = select(func.count(Face.id)).where(
            Face.state == FaceState.UNIDENTIFIED,
            Face.deleted_at.is_(None),
            Face.image_id == self._priority_image_id,
        )
        if self._skipped:
            q = q.where(Face.id.not_in(list(self._skipped)))
        if (session.scalar(q) or 0) == 0:
            self._priority_image_id = None

    def _count_total_unidentified(self, session: "Session") -> int:
        return (
            session.scalar(
                select(func.count(Face.id)).where(
                    Face.state == FaceState.UNIDENTIFIED,
                    Face.deleted_at.is_(None),
                )
            )
            or 0
        )

    def _build_next_face_stmt(self):
        stmt = (
            select(Face)
            .join(Face.image)
            .outerjoin(Image.metadata_rel)
            .options(contains_eager(Face.image).contains_eager(Image.metadata_rel))
            .where(
                Face.state == FaceState.UNIDENTIFIED,
                Face.deleted_at.is_(None),
            )
            .order_by(
                ImageMetadata.taken_at.asc().nulls_last(),
                Image.indexed_at.asc(),
            )
            .limit(1)
        )
        if self._priority_image_id is not None:
            stmt = stmt.where(Face.image_id == self._priority_image_id)
        elif self._skipped_images:
            stmt = stmt.where(Face.image_id.not_in(list(self._skipped_images)))
        if self._skipped:
            stmt = stmt.where(Face.id.not_in(list(self._skipped)))
        return stmt

    def _extract_next_face_data(self, session: "Session") -> Optional[_FaceData]:
        face = (
            session.execute(self._build_next_face_stmt()).unique().scalar_one_or_none()
        )
        if face is None:
            return None
        face_id = face.id
        return _FaceData(
            face_id=face_id,
            crop_path=self.paths.face_crops_dir / f"{face_id}.jpg",
            image_path=Path(face.image.file_path),
            bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
            faiss_id=face.faiss_id,
        )

    # ------------------------------------------------------------------
    # Private: person list
    # ------------------------------------------------------------------

    def _load_persons(self) -> None:
        """Load all persons from DB, preserving filter text and current selection."""
        prev_person_id = (
            self._selected_person.id if self._selected_person is not None else None
        )
        self._selected_person = None
        self._selected_cluster = None
        with self.session_factory() as session:
            persons = (
                session.execute(
                    select(Person)
                    .options(selectinload(Person.clusters))
                    .order_by(Person.name)
                )
                .scalars()
                .all()
            )
            session.expunge_all()
            self._all_persons = list(persons)
        # Preserve the current filter text so the user does not have to retype
        # after each face advance.
        self._filter_persons(self._search_edit.text())
        # Restore the previously selected person if it is still in the filtered
        # list (useful when labelling many faces for the same person in a row).
        if prev_person_id is not None:
            for i in range(self._person_list.count()):
                item = self._person_list.item(i)
                if item is None:  # pragma: no cover
                    continue  # pragma: no cover
                if item.data(QtCore.Qt.ItemDataRole.UserRole).id == prev_person_id:
                    self._person_list.setCurrentItem(item)
                    break
        self._update_confirm_button()

    def _populate_person_list(self, persons: list[Person]) -> None:
        self._person_list.clear()
        for person in persons:
            item = QtWidgets.QListWidgetItem(person.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, person)
            self._person_list.addItem(item)

    def _filter_persons(self, text: str) -> None:
        needle = text.lower().strip()
        if needle:
            filtered = [p for p in self._all_persons if needle in p.name.lower()]
        else:
            filtered = self._all_persons
        self._populate_person_list(filtered)

    def _on_person_selected(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        self._selected_cluster = None

        if current is None:
            self._selected_person = None
            self._clear_cluster_table()
        else:
            person: Person = current.data(QtCore.Qt.ItemDataRole.UserRole)
            self._selected_person = person
            cluster_by_age: dict[str, EmbeddingCluster] = {
                c.age_group: c for c in person.clusters if c.age_group is not None
            }
            scores = (
                self._compute_cluster_scores(cluster_by_age)
                if self._query_embedding is not None
                else {}
            )
            best_row = self._populate_cluster_table(cluster_by_age, scores)
            if best_row is not None:
                self._cluster_table.selectRow(best_row)

        self._update_confirm_button()

    def _clear_cluster_table(self) -> None:
        """Reset all cluster table rows to empty state with no associated data."""
        self._cluster_table.blockSignals(True)
        self._cluster_table.clearSelection()
        for row in range(self._cluster_table.rowCount()):
            name_item = self._cluster_table.item(row, _COL_NAME)
            score_item = self._cluster_table.item(row, _COL_SCORE)
            if name_item is not None:
                name_item.setData(QtCore.Qt.ItemDataRole.UserRole, None)
            if score_item is not None:
                score_item.setText("\u2014")
        self._cluster_table.blockSignals(False)

    def _populate_cluster_table(
        self,
        cluster_by_age: dict[str, EmbeddingCluster],
        scores: dict[str, float],
    ) -> Optional[int]:
        """Fill cluster table rows with cluster data and similarity scores.

        Returns the index of the best-scoring row, or None if no scores are available.
        """
        best_row: Optional[int] = None
        best_score: float = -1.0
        self._cluster_table.blockSignals(True)
        self._cluster_table.clearSelection()
        for row, age_key in enumerate(AGE_CLUSTERS):
            cluster = cluster_by_age.get(age_key)
            name_item = self._cluster_table.item(row, _COL_NAME)
            score_item = self._cluster_table.item(row, _COL_SCORE)
            if name_item is None or score_item is None:  # pragma: no cover
                continue  # pragma: no cover
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, cluster)
            if age_key in scores:
                score = scores[age_key]
                score_item.setText(f"{score:.3f}")
                if score > best_score:
                    best_score = score
                    best_row = row
            else:
                score_item.setText("\u2014")
        self._cluster_table.blockSignals(False)
        return best_row

    def _compute_cluster_scores(
        self, cluster_by_age: dict[str, EmbeddingCluster]
    ) -> dict[str, float]:
        """Compute cosine similarity between query_embedding and each cluster mean.

        Only clusters with ≥1 identified face are included.
        """
        assert self._query_embedding is not None

        q = self._query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return {}
        q = q / q_norm

        scores: dict[str, float] = {}
        for age_key, cluster in cluster_by_age.items():
            faiss_ids = self._get_cluster_faiss_ids(cluster.id)
            if not faiss_ids:
                continue
            embeddings = []
            for fid in faiss_ids:
                try:
                    embeddings.append(self.vector_store.get_embedding(fid))
                except Exception:
                    logger.warning(
                        "Failed to retrieve embedding %s", fid, exc_info=True
                    )
            if not embeddings:
                continue
            mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm < 1e-9:
                continue
            mean_vec = mean_vec / mean_norm
            scores[age_key] = float(np.dot(q, mean_vec))

        return scores

    def _get_cluster_faiss_ids(self, cluster_id: int) -> list[int]:
        """Load faiss_ids of all identified faces belonging to a cluster."""
        with self.session_factory() as session:
            rows = (
                session.execute(
                    select(Face.faiss_id).where(
                        Face.cluster_id == cluster_id,
                        Face.state == FaceState.IDENTIFIED,
                        Face.deleted_at.is_(None),
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    def _on_cluster_row_selected(self) -> None:
        selected = self._cluster_table.selectedItems()
        if not selected:
            self._selected_cluster = None
        else:
            row = self._cluster_table.currentRow()
            name_item = self._cluster_table.item(row, _COL_NAME)
            if name_item is not None:
                self._selected_cluster = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
            else:  # pragma: no cover
                self._selected_cluster = None  # pragma: no cover
        self._update_confirm_button()

    def _create_new_person(self) -> None:
        dlg = NewPersonDialog(self.session_factory, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_person_id = dlg.created_person_id()
        if new_person_id is None:
            return

        with self.session_factory() as session:
            loaded = session.execute(
                select(Person)
                .where(Person.id == new_person_id)
                .options(selectinload(Person.clusters))
            ).scalar_one()
            session.expunge_all()

        bisect.insort(self._all_persons, loaded, key=lambda p: p.name)
        self._filter_persons(self._search_edit.text())

        for i in range(self._person_list.count()):
            item = self._person_list.item(i)
            if item is None:  # pragma: no cover
                continue  # pragma: no cover
            p: Person = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if p.id == new_person_id:
                self._person_list.setCurrentItem(item)
                break

    def _update_confirm_button(self) -> None:
        self.confirm_btn.setEnabled(
            self._selected_person is not None and self._selected_cluster is not None
        )

    # ------------------------------------------------------------------
    # Private: UI helpers
    # ------------------------------------------------------------------

    def _show_empty_state(self) -> None:
        with self.session_factory() as session:
            total_unidentified = self._count_total_unidentified(session)
        if total_unidentified == 0:
            msg = self.tr("All done! No unidentified faces remain.")
        else:
            msg = self.tr(
                "All remaining faces skipped this session. "
                "Restart the app to review them again."
            )
        self._image_preview.clear()
        self._crop_label.setPixmap(QtGui.QPixmap())
        self._crop_label.setText(msg)
        self._person_list.clear()
        self._selected_person = None
        self._selected_cluster = None
        self._clear_cluster_table()
        self._set_buttons_enabled(False)
        self._update_confirm_button()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.skip_image_btn.setEnabled(enabled)
        self.skip_btn.setEnabled(enabled)
        self.anonymous_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Private: actions
    # ------------------------------------------------------------------

    def _on_confirm(self) -> None:
        """Assign the current face to the selected person/cluster."""
        if self._current_face_id is None:
            return
        if self._selected_person is None or self._selected_cluster is None:
            return
        person = self._selected_person
        cluster = self._selected_cluster
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is not None:
                face.state = FaceState.IDENTIFIED
                face.person_id = person.id
                face.cluster_id = cluster.id
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        self._load_next_face()

    def _on_cancel(self) -> None:
        """Clear the current person/cluster selection."""
        self._selected_person = None
        self._selected_cluster = None
        self._person_list.setCurrentRow(-1)
        self._clear_cluster_table()
        self._update_confirm_button()

    def _mark_anonymous(self) -> None:
        if self._current_face_id is None:
            return
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is not None:
                face.state = FaceState.ANONYMOUS
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        self._load_next_face()

    def _skip_face(self) -> None:
        if self._current_face_id is None:
            return
        self._skipped.add(self._current_face_id)
        self._load_next_face()

    def _skip_image(self) -> None:
        if self._current_face_id is None:
            return
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is None:
                return
            image_id = face.image_id
            self._skipped_images.add(image_id)
            face_ids = list(
                session.scalars(
                    select(Face.id).where(
                        Face.image_id == image_id,
                        Face.state == FaceState.UNIDENTIFIED,
                        Face.deleted_at.is_(None),
                    )
                )
            )
            self._skipped.update(face_ids)
        self._load_next_face()
