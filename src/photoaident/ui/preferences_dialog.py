from PySide6 import QtWidgets

from photoaident.ui.widgets.filepath_date_widget import FilepathDateWidget


class PreferencesDialog(QtWidgets.QDialog):
    """Dialog to edit application preferences."""

    def __init__(
        self,
        collection_path: str,
        image_count: int = 0,
        face_count: int = 0,
        filepath_date_enabled: bool = False,
        filepath_date_pattern: str = "",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.image_count = image_count
        self.face_count = face_count
        self.setWindowTitle(self.tr("Preferences"))
        self.setMinimumWidth(500)

        layout = QtWidgets.QVBoxLayout(self)

        # Photo Collection Path setting
        group = QtWidgets.QGroupBox(self.tr("Photo Collection"), self)
        group_layout = QtWidgets.QHBoxLayout(group)

        self.path_edit = QtWidgets.QLineEdit(collection_path, self)
        self.path_edit.setPlaceholderText(self.tr("Select photo collection folder..."))

        browse_button = QtWidgets.QPushButton(self.tr("Browse..."), self)
        browse_button.clicked.connect(self._browse_path)

        group_layout.addWidget(QtWidgets.QLabel(self.tr("Path:")))
        group_layout.addWidget(self.path_edit)
        group_layout.addWidget(browse_button)

        layout.addWidget(group)

        # Date from File Path setting
        self._filepath_date_widget = FilepathDateWidget(
            filepath_date_enabled, filepath_date_pattern, self
        )
        layout.addWidget(self._filepath_date_widget)
        layout.addStretch()

        # OK / Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _browse_path(self) -> None:
        """Open a directory selection dialog."""
        current_path = self.path_edit.text()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Photo Collection Folder"), current_path
        )
        if directory:
            self.path_edit.setText(directory)

    def accept(self) -> None:
        """Validate settings before closing. Block if the pattern is invalid."""
        if not self._filepath_date_widget.validate():
            return
        super().accept()

    def get_collection_path(self) -> str:
        """Return the current collection path from the dialog."""
        return self.path_edit.text()

    def is_filepath_date_enabled(self) -> bool:
        """Return whether filepath date extraction is enabled."""
        return self._filepath_date_widget.is_enabled()

    def get_filepath_date_pattern(self) -> str:
        """Return the configured filepath date pattern."""
        return self._filepath_date_widget.pattern()
