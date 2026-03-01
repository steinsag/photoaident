from PySide6 import QtCore

from photoaident.db.database import AGE_CLUSTERS, EmbeddingCluster
from photoaident.ui.widgets.cluster_table_widget import ClusterTableWidget


def _make_cluster(age_group: str, cluster_id: int = 1) -> EmbeddingCluster:
    """Return an unsaved EmbeddingCluster with the given age group."""
    c = EmbeddingCluster(age_group=age_group)
    c.id = cluster_id
    return c


def _single_cluster_map(age_group: str = "adult") -> dict[str, EmbeddingCluster]:
    """Return a {age_group: cluster} dict with one entry."""
    return {age_group: _make_cluster(age_group)}


# ===========================================================================
# populate()
# ===========================================================================


def test_populate_stores_cluster_in_user_role(qtbot):
    """populate() stores the EmbeddingCluster in each row's UserRole data."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    cluster_by_age = _single_cluster_map("adult")

    widget.populate(cluster_by_age, {})

    adult_row = AGE_CLUSTERS.index("adult")
    name_item = widget._table.item(adult_row, 0)
    assert name_item is not None
    stored = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
    assert stored is cluster_by_age["adult"]


def test_populate_sets_score_text_for_scored_cluster(qtbot):
    """populate() writes the formatted score into the score column."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    cluster_by_age = _single_cluster_map("adult")

    widget.populate(cluster_by_age, {"adult": 0.875})

    adult_row = AGE_CLUSTERS.index("adult")
    score_item = widget._table.item(adult_row, 1)
    assert score_item is not None
    assert score_item.text() == "0.875"


def test_populate_sets_dash_for_unscored_cluster(qtbot):
    """populate() writes '—' into the score column for clusters without a score."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)

    widget.populate(_single_cluster_map("adult"), {})

    adult_row = AGE_CLUSTERS.index("adult")
    score_item = widget._table.item(adult_row, 1)
    assert score_item is not None
    assert score_item.text() == "\u2014"


def test_populate_returns_best_row_index(qtbot):
    """populate() returns the row index corresponding to the highest score."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    cluster_by_age = {
        "adult": _make_cluster("adult", 1),
        "senior": _make_cluster("senior", 2),
    }

    best_row = widget.populate(cluster_by_age, {"adult": 0.5, "senior": 0.9})

    assert best_row == AGE_CLUSTERS.index("senior")


def test_populate_returns_none_when_no_scores(qtbot):
    """populate() returns None when the scores dict is empty."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)

    best_row = widget.populate(_single_cluster_map("adult"), {})

    assert best_row is None


# ===========================================================================
# clear_data()
# ===========================================================================


def test_clear_data_resets_scores_to_dash(qtbot):
    """clear_data() resets every score cell to '—'."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    widget.populate(_single_cluster_map("adult"), {"adult": 0.7})

    widget.clear_data()

    for row in range(widget._table.rowCount()):
        score_item = widget._table.item(row, 1)
        assert score_item is not None
        assert score_item.text() == "\u2014"


def test_clear_data_removes_user_role_data(qtbot):
    """clear_data() clears UserRole data from every name cell."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    widget.populate(_single_cluster_map("adult"), {})

    widget.clear_data()

    for row in range(widget._table.rowCount()):
        name_item = widget._table.item(row, 0)
        assert name_item is not None
        assert name_item.data(QtCore.Qt.ItemDataRole.UserRole) is None


# ===========================================================================
# cluster_selected signal
# ===========================================================================


def test_cluster_selected_signal_emits_cluster_on_row_select(qtbot):
    """cluster_selected emits the stored EmbeddingCluster when a row is selected."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    cluster = _make_cluster("adult")
    widget.populate({"adult": cluster}, {})

    received: list = []
    widget.cluster_selected.connect(received.append)

    adult_row = AGE_CLUSTERS.index("adult")
    widget.select_row(adult_row)

    assert len(received) == 1
    assert received[0] is cluster


def test_cluster_selected_signal_emits_none_on_deselect(qtbot):
    """cluster_selected emits None when the table selection is cleared."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    widget.populate(_single_cluster_map("adult"), {})
    adult_row = AGE_CLUSTERS.index("adult")
    widget.select_row(adult_row)

    received: list = []
    widget.cluster_selected.connect(received.append)

    widget._table.clearSelection()

    assert len(received) == 1
    assert received[0] is None


def test_select_row_triggers_signal(qtbot):
    """select_row() emits cluster_selected with the cluster at that row."""
    widget = ClusterTableWidget()
    qtbot.addWidget(widget)
    cluster = _make_cluster("infant")
    widget.populate({"infant": cluster}, {})

    infant_row = AGE_CLUSTERS.index("infant")
    with qtbot.waitSignal(widget.cluster_selected) as blocker:
        widget.select_row(infant_row)

    assert blocker.args[0] is cluster
