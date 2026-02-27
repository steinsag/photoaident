from importlib.metadata import PackageNotFoundError, version

from PySide6 import QtCore, QtWidgets

_APP_VERSION = "unknown"
try:
    _APP_VERSION = version("photoaident")
except PackageNotFoundError:
    pass


class AboutDialog(QtWidgets.QDialog):
    """About dialog showing author, license, and library attributions."""

    _LIBRARIES: list[tuple[str, str]] = [
        ("InsightFace", "https://www.insightface.ai/"),
        ("FAISS", "https://faiss.ai/"),
        ("PySide6", "https://doc.qt.io/qtforpython-6/"),
        ("ONNX Runtime", "https://onnxruntime.ai/"),
        ("SQLAlchemy", "https://www.sqlalchemy.org/"),
        ("Alembic", "https://alembic.sqlalchemy.org/"),
        ("ExifRead", "https://github.com/ianare/exif-py"),
        ("Pillow", "https://python-pillow.github.io/"),
        ("OpenCV", "https://opencv.org/"),
        ("NumPy", "https://numpy.org/"),
    ]

    _ICONS: list[tuple[str, str]] = [
        ("Wikimedia", "https://commons.wikimedia.org/"),
        ("SVG Repo", "https://www.svgrepo.com/"),
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("About PhotoAIdent"))
        self.setMinimumWidth(480)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # App title
        title_label = QtWidgets.QLabel(f"<h2>PhotoAIdent {_APP_VERSION}</h2>")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QtWidgets.QLabel(
            self.tr(
                "Local, privacy-first desktop app for AI-powered"
                " face recognition and photo search."
            )
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        layout.addSpacing(4)

        # Author / info block
        info_html = (
            "<b>"
            + self.tr("Author:")
            + '</b> <a href="mailto:seb.kde@hpfsc.de">Sebastian Stein</a><br>'
            + "<b>"
            + self.tr("Homepage:")
            + '</b> <a href="https://github.com/steinsag/photoaident">'
            + "github.com/steinsag/photoaident</a><br>"
            + "<b>"
            + self.tr("License:")
            + '</b> <a href="https://www.apache.org/licenses/LICENSE-2.0">'
            + "Apache 2.0</a>"
            + " \u2014 "
            + self.tr("Open Source")
        )
        info_label = QtWidgets.QLabel(info_html)
        info_label.setOpenExternalLinks(True)
        info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        layout.addSpacing(4)

        # Separator
        separator_top = QtWidgets.QFrame()
        separator_top.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator_top.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator_top)

        # Libraries header
        libs_header_label = QtWidgets.QLabel(
            "<b>" + self.tr("Open Source Libraries Used") + "</b>"
        )
        libs_header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(libs_header_label)

        # Libraries list — clickable links separated by middle dots
        lib_links = " &nbsp;&middot;&nbsp; ".join(
            f'<a href="{url}">{name}</a>' for name, url in self._LIBRARIES
        )
        libs_label = QtWidgets.QLabel(lib_links)
        libs_label.setOpenExternalLinks(True)
        libs_label.setWordWrap(True)
        libs_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(libs_label)

        layout.addSpacing(4)

        # Separator
        separator_bottom = QtWidgets.QFrame()
        separator_bottom.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator_bottom.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(separator_bottom)

        # Icons header
        icons_header_label = QtWidgets.QLabel(
            "<b>" + self.tr("Icon Sources Used") + "</b>"
        )
        icons_header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icons_header_label)

        # Icons source list - clickable links separated by middle dots
        icon_links = " &nbsp;&middot;&nbsp; ".join(
            f'<a href="{url}">{name}</a>' for name, url in self._ICONS
        )
        icons_label = QtWidgets.QLabel(icon_links)
        icons_label.setOpenExternalLinks(True)
        icons_label.setWordWrap(True)
        icons_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icons_label)

        layout.addSpacing(4)

        # Close button
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, self
        )
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
