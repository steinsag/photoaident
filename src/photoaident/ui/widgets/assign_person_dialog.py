from typing import Optional, TYPE_CHECKING

from PySide6 import QtCore, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from photoaident.db.database import EmbeddingCluster, Person

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker


class AssignPersonDialog(QtWidgets.QDialog):
    """Dialog for assigning a face to a person and embedding cluster.

    Returns (Person, EmbeddingCluster) via result_person_cluster() on accept.
    Creates new Person / EmbeddingCluster records as needed.
    """

    def __init__(
        self,
        session_factory: "sessionmaker",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
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

        # --- Cluster group (hidden until person with >1 cluster selected) ---
        self.cluster_group = QtWidgets.QGroupBox(self.tr("Cluster"))
        cluster_layout = QtWidgets.QVBoxLayout(self.cluster_group)

        self.cluster_list = QtWidgets.QListWidget()
        self.cluster_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.cluster_list.currentItemChanged.connect(self._on_cluster_selected)
        cluster_layout.addWidget(self.cluster_list)

        self.new_cluster_btn = QtWidgets.QPushButton(self.tr("New cluster\u2026"))
        self.new_cluster_btn.clicked.connect(self._create_new_cluster)
        cluster_layout.addWidget(self.new_cluster_btn)

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
        previous: Optional[QtWidgets.QListWidgetItem],
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

        clusters = list(person.clusters)
        self.cluster_list.clear()
        for cluster in clusters:
            label = cluster.label or self.tr("(unlabelled)")
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, cluster)
            self.cluster_list.addItem(item)

        if len(clusters) == 1:
            # Auto-select single cluster; cluster group stays hidden
            self._selected_cluster = clusters[0]
            self.cluster_group.setVisible(False)
        elif len(clusters) > 1:
            self.cluster_group.setVisible(True)
        else:
            # No clusters (edge case)
            self.cluster_group.setVisible(False)

        self._update_ok_button()

    def _on_cluster_selected(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        if current is None:
            self._selected_cluster = None
        else:
            self._selected_cluster = current.data(QtCore.Qt.ItemDataRole.UserRole)
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
            cluster = EmbeddingCluster(person_id=new_person_id)
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

    def _create_new_cluster(self) -> None:
        if self._selected_person is None:
            return

        label_text, ok = QtWidgets.QInputDialog.getText(
            self, self.tr("New Cluster"), self.tr("Label (optional):")
        )
        if not ok:
            return

        label: Optional[str] = label_text.strip() or None
        new_cluster_id: int
        with self.session_factory() as session:
            cluster = EmbeddingCluster(person_id=self._selected_person.id, label=label)
            session.add(cluster)
            session.flush()
            new_cluster_id = cluster.id
            session.commit()

        # Reload person with updated clusters
        person_id = self._selected_person.id
        with self.session_factory() as session:
            loaded = session.execute(
                select(Person)
                .where(Person.id == person_id)
                .options(selectinload(Person.clusters))
            ).scalar_one()
            session.expunge_all()

        self._selected_person = loaded
        for i, p in enumerate(self._all_persons):
            if p.id == person_id:
                self._all_persons[i] = loaded
                break

        # Repopulate cluster list
        self.cluster_list.clear()
        for c in loaded.clusters:
            lbl = c.label or self.tr("(unlabelled)")
            item = QtWidgets.QListWidgetItem(lbl)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, c)
            self.cluster_list.addItem(item)

        self.cluster_group.setVisible(True)

        # Select the new cluster
        for i in range(self.cluster_list.count()):
            citem = self.cluster_list.item(i)
            if citem is None:
                continue
            c: EmbeddingCluster = citem.data(QtCore.Qt.ItemDataRole.UserRole)
            if c.id == new_cluster_id:
                self.cluster_list.setCurrentItem(citem)
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
