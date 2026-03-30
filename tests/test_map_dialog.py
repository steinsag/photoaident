from unittest.mock import patch

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


# --- initial_bbox forwarded to MapWidget ---


def test_initial_bbox_forwarded_to_map_widget(map_dialog_with_bbox):
    """When initial_bbox is provided, set_initial_bbox is called on the map widget."""
    # Verified by the pending op buffered because QML is not yet ready in tests.
    assert len(map_dialog_with_bbox._map_widget._pending_ops) >= 1


def test_no_initial_bbox_leaves_pending_ops_empty(map_dialog):
    """No pending ops are buffered when no initial_bbox is provided."""
    assert map_dialog._map_widget._pending_ops == []


# --- done (geometry save) ---


def test_done_saves_widget_geometry(map_dialog):
    """done() calls save_widget_geometry before closing."""
    with patch("photoaident.ui.widgets.map_dialog.save_widget_geometry") as mock_save:
        map_dialog.done(QtWidgets.QDialog.DialogCode.Accepted)

    mock_save.assert_called_once_with(map_dialog, map_dialog._paths.window_state_file)
