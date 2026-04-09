from PySide6 import QtCore, QtWidgets

from photoaident.db.database import AGE_CLUSTERS, EmbeddingCluster

_COL_NAME = 0
_COL_SCORE = 1


class ClusterTableWidget(QtWidgets.QWidget):
    """5-row age-group table for selecting the embedding cluster."""

    cluster_selected = QtCore.Signal(object)  # emits Optional[EmbeddingCluster]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QtWidgets.QTableWidget(5, 2)
        self._table.setHorizontalHeaderLabels(
            [self.tr("Age Group"), self.tr("Similarity")]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_SCORE, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self._table.setColumnWidth(_COL_SCORE, 80)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        age_display_names = [
            self.tr("Infant (0\u20133)"),
            self.tr("Youngster (4\u201312)"),
            self.tr("Teenager (13\u201319)"),
            self.tr("Adult (20\u201375)"),
            self.tr("Senior (75+)"),
        ]
        for row, display_name in enumerate(age_display_names):
            name_item = QtWidgets.QTableWidgetItem(display_name)
            self._table.setItem(row, _COL_NAME, name_item)
            score_item = QtWidgets.QTableWidgetItem("\u2014")
            score_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, _COL_SCORE, score_item)

        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(
        self,
        cluster_by_age: dict[str, EmbeddingCluster],
        scores: dict[str, float],
    ) -> int | None:
        """Fill rows with cluster data and similarity scores.

        Returns the index of the best-scoring row, or None if no scores available.
        """
        best_row: int | None = None
        best_score: float = -1.0
        self._table.blockSignals(True)
        self._table.clearSelection()
        for row, age_key in enumerate(AGE_CLUSTERS):
            cluster = cluster_by_age.get(age_key)
            name_item = self._table.item(row, _COL_NAME)
            score_item = self._table.item(row, _COL_SCORE)
            if name_item is None or score_item is None:  # pragma: no cover
                continue  # pragma: no cover
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, cluster)
            if age_key in scores:
                score = scores[age_key]
                score_item.setText(f"{score:.0%}")
                if score > best_score:
                    best_score = score
                    best_row = row
            else:
                score_item.setText("\u2014")
        self._table.blockSignals(False)
        return best_row

    def clear_data(self) -> None:
        """Reset all rows to empty state (no cluster data, score reset to —)."""
        self._table.blockSignals(True)
        self._table.clearSelection()
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, _COL_NAME)
            score_item = self._table.item(row, _COL_SCORE)
            if name_item is not None:
                name_item.setData(QtCore.Qt.ItemDataRole.UserRole, None)
            if score_item is not None:
                score_item.setText("\u2014")
        self._table.blockSignals(False)

    def select_row(self, row: int) -> None:
        """Programmatically select a row, emitting cluster_selected."""
        self._table.selectRow(row)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        selected = self._table.selectedItems()
        if not selected:
            self.cluster_selected.emit(None)
        else:
            row = self._table.currentRow()
            name_item = self._table.item(row, _COL_NAME)
            if name_item is not None:
                self.cluster_selected.emit(
                    name_item.data(QtCore.Qt.ItemDataRole.UserRole)
                )
            else:  # pragma: no cover
                self.cluster_selected.emit(None)  # pragma: no cover
