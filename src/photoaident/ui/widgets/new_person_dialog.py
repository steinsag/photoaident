from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from photoaident.db.database import AGE_CLUSTERS, EmbeddingCluster, Person

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker


class NewPersonDialog(QtWidgets.QDialog):
    """Dialog for creating a new person with all 5 age-group clusters.

    Shows a name input field and standard OK/Cancel buttons.
    The OK button is disabled until a non-empty name is entered.
    On acceptance, creates a Person and 5 EmbeddingCluster rows in the DB.
    Use created_person_id() to retrieve the new person's id after exec().
    """

    def __init__(
        self,
        session_factory: "sessionmaker",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_factory = session_factory
        self._created_person_id: int | None = None
        self.setWindowTitle(self.tr("New Person"))
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form_layout = QtWidgets.QFormLayout()
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.textChanged.connect(self._on_name_changed)
        form_layout.addRow(self.tr("Name:"), self._name_edit)
        layout.addLayout(form_layout)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        self._update_ok_button()

    def _on_name_changed(self, _text: str) -> None:
        self._update_ok_button()

    def _update_ok_button(self) -> None:
        ok_btn = self._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(bool(self._name_edit.text().strip()))

    def _persist_new_person(self, name: str) -> int:
        """Write Person + 5 EmbeddingCluster rows to DB; return the new person id."""
        with self.session_factory() as session:
            person = Person(name=name)
            session.add(person)
            session.flush()
            person_id = person.id
            for age_key in AGE_CLUSTERS:
                session.add(
                    EmbeddingCluster(
                        person_id=person_id,
                        label=age_key,
                        age_group=age_key,
                    )
                )
            session.commit()
        return person_id

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        self._created_person_id = self._persist_new_person(name)
        self.accept()

    def created_person_id(self) -> int | None:
        """Return the new person's DB id, or None if not accepted."""
        return self._created_person_id
