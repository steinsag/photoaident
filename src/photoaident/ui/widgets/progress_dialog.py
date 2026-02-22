from PySide6 import QtWidgets, QtCore


class ProgressDialog(QtWidgets.QDialog):
    """
    Modal dialog to show progress of a background task.
    """

    def __init__(
        self, title: str, message: str, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(300)
        self.setModal(True)

        # Prevent manual closing
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowCloseButtonHint
        )

        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(message)
        layout.addWidget(self.label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate at first
        layout.addWidget(self.progress_bar)

    @QtCore.Slot(str)
    def update_status(self, text: str):
        self.label.setText(text)

    @QtCore.Slot(int, int)
    def update_progress(self, current: int, total: int):
        if self.progress_bar.maximum() != total:
            self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
