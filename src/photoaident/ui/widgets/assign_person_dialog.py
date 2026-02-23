from typing import Optional, TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from photoaident.db.database import (
    AGE_CLUSTERS,
    EmbeddingCluster,
    Face,
    FaceState,
    Person,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from photoaident.db.vector_store import VectorStore

_COL_INDICATOR = 0
_COL_NAME = 1
_COL_SCORE = 2


class AssignPersonDialog(QtWidgets.QDialog):
    """Dialog for assigning a face to a person and age-group embedding cluster.

    Returns (Person, EmbeddingCluster) via result_person_cluster() on accept.
    Creates new Person records (with all 5 age clusters) as needed.
    """

    def __init__(
        self,
        session_factory: "sessionmaker",
        query_embedding: Optional[np.ndarray] = None,
        vector_store: Optional["VectorStore"] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
        self.query_embedding = query_embedding
        self.vector_store = vector_store
        self._selected_person: Optional[Person] = None
        self._selected_cluster: Optional[EmbeddingCluster] = None
        self._all_persons: list[Person] = []

        self.setWindowTitle(self.tr("Assign to Person"))
        self.setMinimumWidth(420)
        self._setup_ui()
        self._load_persons()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # --- Person group ---
        person_group = QtWidgets.QGroupBox(self.tr("Person"))
        person_layout = QtWidgets.QVBoxLayout(person_group)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText(self.tr("Search\u2026"))
        self.search_edit.textChanged.connect(self._filter_persons)
        person_layout.addWidget(self.search_edit)

        self.person_list = QtWidgets.QListWidget()
        self.person_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.person_list.currentItemChanged.connect(self._on_person_selected)
        person_layout.addWidget(self.person_list)

        self.new_person_btn = QtWidgets.QPushButton(self.tr("New person\u2026"))
        self.new_person_btn.clicked.connect(self._create_new_person)
        person_layout.addWidget(self.new_person_btn)

        layout.addWidget(person_group)

        # --- Cluster group (hidden until person selected) ---
        self.cluster_group = QtWidgets.QGroupBox(self.tr("Age Group"))
        cluster_layout = QtWidgets.QVBoxLayout(self.cluster_group)

        # 5 rows × 3 columns: indicator | name | similarity
        self.cluster_table = QtWidgets.QTableWidget(5, 3)
        self.cluster_table.setHorizontalHeaderLabels(
            ["", self.tr("Age Group"), self.tr("Similarity")]
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            _COL_INDICATOR, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            _COL_SCORE, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self.cluster_table.setColumnWidth(_COL_SCORE, 80)
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.cluster_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.cluster_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        # Pre-fill age-group name column
        age_display_names = [
            self.tr("Infant (0\u20133)"),
            self.tr("Youngster (4\u201312)"),
            self.tr("Teenager (13\u201319)"),
            self.tr("Adult (20\u201375)"),
            self.tr("Senior (75+)"),
        ]
        for row, display_name in enumerate(age_display_names):
            indicator_item = QtWidgets.QTableWidgetItem("")
            indicator_item.setFlags(
                indicator_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            self.cluster_table.setItem(row, _COL_INDICATOR, indicator_item)
            name_item = QtWidgets.QTableWidgetItem(display_name)
            self.cluster_table.setItem(row, _COL_NAME, name_item)
            score_item = QtWidgets.QTableWidgetItem("\u2014")
            score_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.cluster_table.setItem(row, _COL_SCORE, score_item)

        self.cluster_table.itemSelectionChanged.connect(self._on_cluster_row_selected)
        cluster_layout.addWidget(self.cluster_table)

        self.cluster_group.setVisible(False)
        layout.addWidget(self.cluster_group)

        # --- Dialog buttons ---
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._update_ok_button()

    def _load_persons(self) -> None:
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
        self._populate_person_list(self._all_persons)

    def _populate_person_list(self, persons: list[Person]) -> None:
        self.person_list.clear()
        for person in persons:
            item = QtWidgets.QListWidgetItem(person.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, person)
            self.person_list.addItem(item)

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
        _previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        if current is None:
            self._selected_person = None
            self._selected_cluster = None
            self.cluster_group.setVisible(False)
            self._update_ok_button()
            return

        person: Person = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._selected_person = person
        self._selected_cluster = None

        # Build mapping from age_group key → cluster
        cluster_by_age: dict[str, EmbeddingCluster] = {
            c.age_group: c for c in person.clusters if c.age_group is not None
        }

        # Compute similarity scores if embedding available
        scores: dict[str, float] = {}
        if self.query_embedding is not None and self.vector_store is not None:
            scores = self._compute_cluster_scores(cluster_by_age)

        # Populate table rows
        best_row: Optional[int] = None
        best_score: float = -1.0
        self.cluster_table.blockSignals(True)
        self.cluster_table.clearSelection()
        for row, age_key in enumerate(AGE_CLUSTERS):
            cluster = cluster_by_age.get(age_key)
            name_item = self.cluster_table.item(row, _COL_NAME)
            score_item = self.cluster_table.item(row, _COL_SCORE)
            if name_item is None or score_item is None:
                continue
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, cluster)
            if age_key in scores:
                score = scores[age_key]
                score_item.setText(f"{score:.3f}")
                if score > best_score:
                    best_score = score
                    best_row = row
            else:
                score_item.setText("\u2014")

        self.cluster_table.blockSignals(False)

        # Pre-select best-scoring row (if any)
        if best_row is not None:
            self.cluster_table.selectRow(best_row)
        # _on_cluster_row_selected will fire and update _selected_cluster

        self.cluster_group.setVisible(True)
        self._update_ok_button()

    def _compute_cluster_scores(
        self, cluster_by_age: dict[str, EmbeddingCluster]
    ) -> dict[str, float]:
        """Compute cosine similarity between query_embedding and each cluster mean.

        Only clusters that have ≥1 identified face are included in the result.
        """
        assert self.query_embedding is not None
        assert self.vector_store is not None

        q = self.query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
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
                except (IndexError, Exception):
                    pass
            if not embeddings:
                continue
            mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm == 0.0:
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
        selected = self.cluster_table.selectedItems()
        if not selected:
            self._selected_cluster = None
        else:
            row = self.cluster_table.currentRow()
            name_item = self.cluster_table.item(row, _COL_NAME)
            if name_item is not None:
                self._selected_cluster = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
            else:
                self._selected_cluster = None
        self._update_ok_button()

    def _create_new_person(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, self.tr("New Person"), self.tr("Name:")
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        new_person_id: int
        with self.session_factory() as session:
            person = Person(name=name)
            session.add(person)
            session.flush()
            new_person_id = person.id
            for age_key in AGE_CLUSTERS:
                cluster = EmbeddingCluster(
                    person_id=new_person_id, label=age_key, age_group=age_key
                )
                session.add(cluster)
            session.commit()

        # Reload with clusters eagerly
        with self.session_factory() as session:
            loaded = session.execute(
                select(Person)
                .where(Person.id == new_person_id)
                .options(selectinload(Person.clusters))
            ).scalar_one()
            session.expunge_all()

        self._all_persons.append(loaded)
        self._filter_persons(self.search_edit.text())

        # Select the new person in the list
        for i in range(self.person_list.count()):
            item = self.person_list.item(i)
            if item is None:
                continue
            p: Person = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if p.id == new_person_id:
                self.person_list.setCurrentItem(item)
                break

    def _update_ok_button(self) -> None:
        ok_btn = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(
                self._selected_person is not None and self._selected_cluster is not None
            )

    def result_person_cluster(
        self,
    ) -> Optional[tuple[Person, EmbeddingCluster]]:
        """Return (Person, EmbeddingCluster) if dialog was accepted, else None."""
        if self._selected_person is not None and self._selected_cluster is not None:
            return self._selected_person, self._selected_cluster
        return None
