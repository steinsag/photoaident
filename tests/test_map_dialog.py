from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtQuickWidgets, QtWidgets

from photoaident.core.geo import GpsBoundingBox
from photoaident.ui.widgets.map_dialog import MapLocationDialog

BBOX_GERMANY = GpsBoundingBox(south=48.0, west=10.0, north=52.0, east=14.0)


def _make_dialog(
    qtbot,
    tmp_app_paths,
    initial_bbox: GpsBoundingBox | None = None,
) -> MapLocationDialog:
    """Create a MapLocationDialog with QML engine startup suppressed."""
    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = MapLocationDialog(tmp_app_paths, initial_bbox=initial_bbox)
    qtbot.addWidget(dialog)
    return dialog


@pytest.fixture
def map_dialog(qtbot, tmp_app_paths):
    """A MapLocationDialog with no initial bbox."""
    return _make_dialog(qtbot, tmp_app_paths)


@pytest.fixture
def map_dialog_with_bbox(qtbot, tmp_app_paths):
    """A MapLocationDialog seeded with an initial bounding box."""
    return _make_dialog(qtbot, tmp_app_paths, initial_bbox=BBOX_GERMANY)


# --- selected_bbox ---


def test_selected_bbox_initially_none(map_dialog):
    """selected_bbox() returns None before the dialog is accepted."""
    assert map_dialog.selected_bbox() is None


# --- _on_accept ---


def test_on_accept_stores_selected_bbox(map_dialog):
    """Stores the bbox returned by map_widget.current_bbox() on accept."""
    expected = GpsBoundingBox(south=1.0, west=2.0, north=3.0, east=4.0)

    with (
        patch.object(map_dialog._map_widget, "current_bbox", return_value=expected),
        patch.object(MapLocationDialog, "accept"),
    ):
        map_dialog._on_accept()

    assert map_dialog.selected_bbox() == expected


def _mock_map_root(dialog: MapLocationDialog) -> MagicMock:
    """Replace the map widget's internal QQuickWidget with a mock root."""
    mock_root = MagicMock()
    dialog._map_widget._quick_widget = MagicMock()
    dialog._map_widget._quick_widget.rootObject.return_value = mock_root
    return mock_root


# --- initial_bbox forwarded to MapWidget ---


def test_initial_bbox_forwarded_to_map_widget(map_dialog_with_bbox):
    """initial_bbox is provided — bbox properties are set on the QML root when ready."""
    mock_root = _mock_map_root(map_dialog_with_bbox)

    map_dialog_with_bbox._map_widget._on_qml_status_changed(
        QtQuickWidgets.QQuickWidget.Status.Ready
    )

    set_props = {c.args[0] for c in mock_root.setProperty.call_args_list}
    assert "pendingBbox" in set_props


def test_no_initial_bbox_leaves_pending_ops_empty(map_dialog):
    """No initial_bbox — no bbox properties are set on the QML root when ready."""
    mock_root = _mock_map_root(map_dialog)

    map_dialog._map_widget._on_qml_status_changed(
        QtQuickWidgets.QQuickWidget.Status.Ready
    )

    set_props = {c.args[0] for c in mock_root.setProperty.call_args_list}
    assert "pendingBbox" not in set_props


# --- done (geometry save) ---


def test_done_saves_widget_geometry(map_dialog):
    """done() calls save_widget_geometry before closing."""
    with patch("photoaident.ui.widgets.map_dialog.save_widget_geometry") as mock_save:
        map_dialog.done(QtWidgets.QDialog.DialogCode.Accepted)

    mock_save.assert_called_once_with(map_dialog, map_dialog._paths.window_state_file)
