from PySide6 import QtWidgets

from photoaident.core.filepath_date import (
    PatternErrorCode,
    PatternValidationError,
    compile_pattern,
)


class FilepathDateWidget(QtWidgets.QWidget):
    """Reusable widget for the 'Date from File Path' settings section.

    Encapsulates the checkbox, instructions label, and pattern field used in
    both :class:`PreferencesDialog` and :class:`OnboardingDialog`.
    """

    def __init__(
        self,
        enabled: bool = False,
        pattern: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        group = QtWidgets.QGroupBox(self.tr("Date from File Path"), self)
        date_layout = QtWidgets.QVBoxLayout(group)

        self._checkbox = QtWidgets.QCheckBox(
            self.tr("Extract dates from file paths when EXIF is missing"), self
        )
        self._checkbox.setChecked(enabled)
        date_layout.addWidget(self._checkbox)

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
        self._pattern_edit = QtWidgets.QLineEdit(pattern, self)
        self._pattern_edit.setPlaceholderText("{YYYY}-{MM}-{DD}")
        pattern_row.addWidget(pattern_label)
        pattern_row.addWidget(self._pattern_edit)
        date_layout.addLayout(pattern_row)

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(group)

        self._set_pattern_enabled(enabled)
        self._checkbox.toggled.connect(self._set_pattern_enabled)

    def _set_pattern_enabled(self, enabled: bool) -> None:
        self._pattern_edit.setEnabled(enabled)

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

    def is_enabled(self) -> bool:
        """Return whether date extraction from file paths is enabled."""
        return self._checkbox.isChecked()

    def pattern(self) -> str:
        """Return the configured pattern, stripped of whitespace."""
        return self._pattern_edit.text().strip()

    def validate(self) -> bool:
        """Validate the current state.

        Returns True if valid or the checkbox is unchecked. When checked with
        an invalid pattern, shows a warning dialog and returns False.
        """
        if not self._checkbox.isChecked():
            return True
        try:
            compile_pattern(self.pattern())
        except PatternValidationError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid Pattern"),
                self._translate_pattern_error(exc.code),
            )
            return False
        return True
