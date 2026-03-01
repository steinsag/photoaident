import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets
from sqlalchemy import func, select
from sqlalchemy.orm import contains_eager, selectinload

from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    ImageMetadata,
    Person,
)
from photoaident.ui.widgets.cluster_table_widget import ClusterTableWidget
from photoaident.ui.widgets.face_crop_widget import FaceCropWidget
from photoaident.ui.widgets.image_preview_widget import ImagePreviewWidget
from photoaident.ui.widgets.new_person_dialog import NewPersonDialog
from photoaident.ui.widgets.person_list_widget import PersonListWidget

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

    from photoaident.db.vector_store import VectorStore
    from photoaident.paths import AppPaths


@dataclass(frozen=True, slots=True)
class FaceData:
    """Data for the currently displayed face."""

    face_id: int
    crop_path: Optional[Path]
    image_path: Path
    bbox: tuple[int, int, int, int]
    faiss_id: int


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

        self._image_preview = ImagePreviewWidget()
        self._image_preview.setMinimumSize(300, 300)
        top_layout.addWidget(self._image_preview, stretch=1)

        self._face_crop = FaceCropWidget()
        top_layout.addWidget(self._face_crop)

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

        self._person_widget = PersonListWidget()
        self._person_widget.person_selected.connect(self._on_person_selected)
        self._person_widget.new_person_requested.connect(self._on_new_person_requested)
        group_layout.addWidget(self._person_widget)

        # Right panel: cluster table + confirm/cancel
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._cluster_widget = ClusterTableWidget()
        self._cluster_widget.cluster_selected.connect(self._on_cluster_selected)
        right_layout.addWidget(self._cluster_widget, stretch=1)

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

    # ------------------------------------------------------------------
    # Backward-compatible aliases for internal widget access (used by tests)
    # ------------------------------------------------------------------

    @property
    def _person_list(self) -> "QtWidgets.QListWidget":
        """Return the inner QListWidget of the person panel."""
        return self._person_widget._list_widget

    @property
    def _search_edit(self) -> "QtWidgets.QLineEdit":
        """Return the search field of the person panel."""
        return self._person_widget._search_edit

    @property
    def _cluster_table(self) -> "QtWidgets.QTableWidget":
        """Return the inner QTableWidget of the cluster panel."""
        return self._cluster_widget._table

    @property
    def _crop_label(self) -> FaceCropWidget:
        """Return the face crop widget."""
        return self._face_crop

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refresh(self, priority_image_id: Optional[int] = None) -> None:
        """Reload count + first face. Called when page becomes visible."""
        self._skipped.clear()
        self._skipped_images.clear()
        self._priority_image_id = priority_image_id
        self._load_persons()
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
        self._face_crop.load(face_data.crop_path)
        self._set_buttons_enabled(True)
        self._refresh_cluster_selection()

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

    def _extract_next_face_data(self, session: "Session") -> Optional[FaceData]:
        face = (
            session.execute(self._build_next_face_stmt()).unique().scalar_one_or_none()
        )
        if face is None:
            return None
        face_id = face.id
        return FaceData(
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
            self._person_widget.set_persons(list(persons))
        # Restore the previously selected person if it is still in the filtered
        # list (useful when labelling many faces for the same person in a row).
        if prev_person_id is not None:
            self._person_widget.select_by_id(prev_person_id)
        self._update_confirm_button()

    def _on_person_selected(
        self,
        person: Optional[Person],
        _: object = None,
    ) -> None:
        """Handle person selection from PersonListWidget signal or direct call."""
        self._selected_person = person
        self._populate_clusters_for_person(person)

    def _populate_clusters_for_person(self, person: Optional[Person]) -> None:
        """Populate the cluster widget for *person*, auto-selecting the best match."""
        self._selected_cluster = None
        if person is None:
            self._cluster_widget.clear_data()
        else:
            cluster_by_age: dict[str, EmbeddingCluster] = {
                c.age_group: c for c in person.clusters if c.age_group is not None
            }
            scores = (
                self._compute_cluster_scores(cluster_by_age)
                if self._query_embedding is not None
                else {}
            )
            best_row = self._cluster_widget.populate(cluster_by_age, scores)
            if best_row is not None:
                self._cluster_widget.select_row(best_row)
        self._update_confirm_button()

    def _refresh_cluster_selection(self) -> None:
        """Recalculate cluster scores for the current face; no DB round-trip."""
        self._populate_clusters_for_person(self._selected_person)

    def _on_cluster_selected(self, cluster: Optional[EmbeddingCluster]) -> None:
        self._selected_cluster = cluster
        self._update_confirm_button()

    def _on_new_person_requested(self) -> None:
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

        self._person_widget.add_person_sorted(loaded)
        self._person_widget.select_by_id(new_person_id)

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
        self._face_crop.clear()
        self._face_crop.setText(msg)
        self._person_widget.set_persons([])
        self._selected_person = None
        self._selected_cluster = None
        self._cluster_widget.clear_data()
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
        self._person_widget.clear_selection()
        self._cluster_widget.clear_data()
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
                self._load_next_face()
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

    # ------------------------------------------------------------------
    # Backward-compatible method aliases
    # ------------------------------------------------------------------

    def _filter_persons(self, text: str) -> None:
        """Apply person filter on the person list widget."""
        self._person_widget._apply_filter(text)

    def _create_new_person(self) -> None:
        """Alias for _on_new_person_requested (used in tests)."""
        self._on_new_person_requested()
