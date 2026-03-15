from PySide6 import QtWidgets

from photoaident.core.filepath_date import (
    PatternErrorCode,
    PatternValidationError,
    compile_pattern,
)


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
        date_group = QtWidgets.QGroupBox(self.tr("Date from File Path"), self)
        date_layout = QtWidgets.QVBoxLayout(date_group)

        self._filepath_date_checkbox = QtWidgets.QCheckBox(
            self.tr("Extract dates from file paths when EXIF is missing"), self
        )
        self._filepath_date_checkbox.setChecked(filepath_date_enabled)
        date_layout.addWidget(self._filepath_date_checkbox)

        instructions = QtWidgets.QLabel(
            self.tr(
                "Use placeholders to describe the date format in your folder names "
                "or file names. Supported placeholders: "
                "{YYYY} (year), {MM} (2-digit month), {M} (1-or-2-digit month), "
                "{DD} (2-digit day), {D} (1-or-2-digit day).\n"
                "Examples: {YYYY}-{MM}-{DD}  ·  {DD}.{M}.{YYYY}  ·  PXL{YYYY}{MM}{DD}"
            ),
            self,
        )
        instructions.setWordWrap(True)
        date_layout.addWidget(instructions)

        pattern_row = QtWidgets.QHBoxLayout()
        pattern_label = QtWidgets.QLabel(self.tr("Pattern:"), self)
        self._filepath_date_pattern_edit = QtWidgets.QLineEdit(
            filepath_date_pattern, self
        )
        self._filepath_date_pattern_edit.setPlaceholderText("{YYYY}-{MM}-{DD}")
        pattern_row.addWidget(pattern_label)
        pattern_row.addWidget(self._filepath_date_pattern_edit)
        date_layout.addLayout(pattern_row)

        layout.addWidget(date_group)
        layout.addStretch()

        # Enable/disable pattern widgets based on checkbox state
        self._set_date_widgets_enabled(filepath_date_enabled)
        self._filepath_date_checkbox.toggled.connect(self._set_date_widgets_enabled)

        # OK / Cancel buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _set_date_widgets_enabled(self, enabled: bool) -> None:
        self._filepath_date_pattern_edit.setEnabled(enabled)

    def _browse_path(self):
        """Open a directory selection dialog."""
        current_path = self.path_edit.text()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, self.tr("Select Photo Collection Folder"), current_path
        )
        if directory:
            self.path_edit.setText(directory)

    def _translate_pattern_error(self, code: PatternErrorCode) -> str:
        """Return a translated error message for the given validation error code."""
        messages: dict[PatternErrorCode, str] = {
            PatternErrorCode.EMPTY: self.tr("Pattern must not be empty."),
            PatternErrorCode.YEAR_MISSING: self.tr(
                "Pattern must contain exactly one {YYYY} placeholder."
            ),
            PatternErrorCode.YEAR_DUPLICATE: self.tr(
                "Pattern must not contain {YYYY} more than once."
            ),
            PatternErrorCode.MONTH_CONFLICT: self.tr(
                "Pattern must not contain both {MM} and {M}."
            ),
            PatternErrorCode.MONTH_MISSING: self.tr(
                "Pattern must contain a month placeholder ({MM} or {M})."
            ),
            PatternErrorCode.MONTH_MM_DUPLICATE: self.tr(
                "Pattern must not contain {MM} more than once."
            ),
            PatternErrorCode.MONTH_M_DUPLICATE: self.tr(
                "Pattern must not contain {M} more than once."
            ),
            PatternErrorCode.DAY_CONFLICT: self.tr(
                "Pattern must not contain both {DD} and {D}."
            ),
            PatternErrorCode.DAY_MISSING: self.tr(
                "Pattern must contain a day placeholder ({DD} or {D})."
            ),
            PatternErrorCode.DAY_DD_DUPLICATE: self.tr(
                "Pattern must not contain {DD} more than once."
            ),
            PatternErrorCode.DAY_D_DUPLICATE: self.tr(
                "Pattern must not contain {D} more than once."
            ),
        }
        return messages[code]

    def accept(self) -> None:
        """Validate settings before closing. Block if the pattern is invalid."""
        if self._filepath_date_checkbox.isChecked():
            pattern = self._filepath_date_pattern_edit.text().strip()
            try:
                compile_pattern(pattern)
            except PatternValidationError as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Invalid Pattern"),
                    self._translate_pattern_error(exc.code),
                )
                return
        super().accept()

    def get_collection_path(self) -> str:
        """Return the current collection path from the dialog."""
        return self.path_edit.text()

    def is_filepath_date_enabled(self) -> bool:
        """Return whether filepath date extraction is enabled."""
        return self._filepath_date_checkbox.isChecked()

    def get_filepath_date_pattern(self) -> str:
        """Return the configured filepath date pattern."""
        return self._filepath_date_pattern_edit.text().strip()
