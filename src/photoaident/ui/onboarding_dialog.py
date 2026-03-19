from PySide6 import QtWidgets

from photoaident.ui.widgets.filepath_date_widget import FilepathDateWidget


class OnboardingDialog(QtWidgets.QDialog):
    """First-run dialog that asks the user to select their photo collection folder.

    Tests should call ``_on_accept()`` directly rather than ``exec()`` to avoid
    native window-system interaction on macOS (see CLAUDE.md testing guidelines).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Welcome to PhotoAIdent"))
        self.setMinimumWidth(500)

        self._selected_path: str = ""

        layout = QtWidgets.QVBoxLayout(self)

        welcome_label = QtWidgets.QLabel(
            self.tr(
                "Welcome to PhotoAIdent!\n\n"
                "To get started, please select your photo collection folder."
            )
        )
        welcome_label.setWordWrap(True)
        layout.addWidget(welcome_label)
        layout.addSpacing(12)

        path_layout = QtWidgets.QHBoxLayout()
        self._path_edit = QtWidgets.QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText(self.tr("No folder selected"))
        browse_btn = QtWidgets.QPushButton(self.tr("Browse..."))
        path_layout.addWidget(self._path_edit)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        layout.addSpacing(12)

        self._filepath_date_widget = FilepathDateWidget(parent=self)
        layout.addWidget(self._filepath_date_widget)
        layout.addSpacing(12)

        button_box = QtWidgets.QDialogButtonBox()
        self._start_btn = button_box.addButton(
            self.tr("Start Indexing"),
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._start_btn.setEnabled(False)
        layout.addWidget(button_box)

        browse_btn.clicked.connect(self._browse)
        button_box.accepted.connect(self._on_accept)

    def _browse(self) -> None:
        """Open a folder picker and populate the path field."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Photo Collection Folder"), ""
        )
        if folder:
            self._path_edit.setText(folder)
            self._start_btn.setEnabled(True)

    def _on_accept(self) -> None:
        """Validate settings, store the chosen path, and accept the dialog."""
        if not self._filepath_date_widget.validate():
            return
        self._selected_path = self._path_edit.text()
        self.accept()

    def selected_path(self) -> str:
        """Return the folder path chosen by the user (empty string if none)."""
        return self._selected_path

    def is_filepath_date_enabled(self) -> bool:
        """Return whether filepath date extraction is enabled."""
        return self._filepath_date_widget.is_enabled()

    def get_filepath_date_pattern(self) -> str:
        """Return the configured filepath date pattern."""
        return self._filepath_date_widget.pattern()
