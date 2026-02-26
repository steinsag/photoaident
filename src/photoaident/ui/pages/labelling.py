from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from sqlalchemy import func, select
from sqlalchemy.orm import contains_eager

from photoaident.db.database import Face, FaceState, Image, ImageMetadata
from photoaident.ui.widgets.assign_person_dialog import AssignPersonDialog
from photoaident.ui.widgets.face_crop import FaceCropWidget

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


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
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.face_crop = FaceCropWidget()
        layout.addWidget(self.face_crop, stretch=1)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.addStretch()

        self.assign_btn = QtWidgets.QPushButton(self.tr("Assign to Person\u2026"))
        self.assign_btn.clicked.connect(self._assign_face)
        action_layout.addWidget(self.assign_btn)

        self.anonymous_btn = QtWidgets.QPushButton(self.tr("Mark Anonymous"))
        self.anonymous_btn.clicked.connect(self._mark_anonymous)
        action_layout.addWidget(self.anonymous_btn)

        self.skip_btn = QtWidgets.QPushButton(self.tr("Skip"))
        self.skip_btn.clicked.connect(self._skip_face)
        action_layout.addWidget(self.skip_btn)

        self.skip_image_btn = QtWidgets.QPushButton(self.tr("Skip Image"))
        self.skip_image_btn.clicked.connect(self._skip_image)
        action_layout.addWidget(self.skip_image_btn)

        action_layout.addStretch()
        layout.addLayout(action_layout)

        self._set_buttons_enabled(False)

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
        face_data: Optional[tuple[int, Path, Optional[Path], str, float]] = None
        with self.session_factory() as session:
            self._maybe_clear_priority(session)
            status_count = self._count_remaining(session)
            total_unidentified = self._count_total_unidentified(session)
            face_data = self._extract_next_face_data(session)

        if face_data is None:
            self._current_face_id = None
            self._show_empty_state(total_unidentified)
            return

        face_id, crop_path, thumb_path, taken_at, confidence = face_data
        self._current_face_id = face_id
        if self._priority_image_id is not None:
            self.status_label.setText(
                self.tr("{count} face(s) remaining in this image").format(
                    count=status_count
                )
            )
        else:
            self.status_label.setText(
                self.tr("{count} face(s) remaining").format(count=status_count)
            )
        self.face_crop.load(
            crop_path=crop_path,
            thumb_path=thumb_path,
            taken_at=taken_at,
            confidence=confidence,
        )
        self._set_buttons_enabled(True)

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

    def _count_remaining(self, session: "Session") -> int:
        if self._priority_image_id is not None:
            q = select(func.count(Face.id)).where(
                Face.state == FaceState.UNIDENTIFIED,
                Face.deleted_at.is_(None),
                Face.image_id == self._priority_image_id,
            )
        else:
            q = select(func.count(Face.id)).where(
                Face.state == FaceState.UNIDENTIFIED,
                Face.deleted_at.is_(None),
            )
            if self._skipped_images:
                q = q.where(Face.image_id.not_in(list(self._skipped_images)))
        if self._skipped:
            q = q.where(Face.id.not_in(list(self._skipped)))
        return session.scalar(q) or 0

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

    def _extract_next_face_data(
        self, session: "Session"
    ) -> Optional[tuple[int, Path, Optional[Path], str, float]]:
        face = (
            session.execute(self._build_next_face_stmt()).unique().scalar_one_or_none()
        )
        if face is None:
            return None
        face_id = face.id
        crop_path = self.paths.face_crops_dir / f"{face_id}.jpg"
        thumb_path = (
            self.paths.thumbs_dir / f"{face.image.file_hash}.jpg"
            if face.image.file_hash
            else None
        )
        meta = face.image.metadata_rel
        taken_at = (
            meta.taken_at.strftime("%Y-%m-%d")
            if meta is not None and meta.taken_at is not None
            else self.tr("Unknown")
        )
        return (face_id, crop_path, thumb_path, taken_at, face.detection_confidence)

    # ------------------------------------------------------------------
    # Private: UI helpers
    # ------------------------------------------------------------------

    def _show_empty_state(self, total_unidentified: int) -> None:
        if total_unidentified == 0:
            msg = self.tr("All done! No unidentified faces remain.")
        else:
            msg = self.tr(
                "All remaining faces skipped this session. "
                "Restart the app to review them again."
            )
        self.status_label.setText(msg)
        self.face_crop.clear()
        self._set_buttons_enabled(False)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.assign_btn.setEnabled(enabled)
        self.anonymous_btn.setEnabled(enabled)
        self.skip_btn.setEnabled(enabled)
        self.skip_image_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Private: actions
    # ------------------------------------------------------------------

    def _assign_face(self) -> None:
        if self._current_face_id is None:
            return
        query_embedding = None
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is not None:
                try:
                    query_embedding = self.vector_store.get_embedding(face.faiss_id)
                except Exception:
                    pass
        dialog = AssignPersonDialog(
            self.session_factory,
            query_embedding=query_embedding,
            vector_store=self.vector_store,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        result = dialog.result_person_cluster()
        if result is None:
            return
        person, cluster = result
        with self.session_factory() as session:
            face = session.get(Face, self._current_face_id)
            if face is not None:
                face.state = FaceState.IDENTIFIED
                face.person_id = person.id
                face.cluster_id = cluster.id
                face.labelled_at = datetime.now(timezone.utc)
                session.commit()
        self._load_next_face()

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
