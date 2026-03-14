import logging
from unittest.mock import MagicMock, patch

import pytest

from photoaident.core.geo import GpsBoundingBox
from photoaident.ui.widgets.map_dialog import MapLocationDialog


def test_build_bbox_valid():
    bbox = MapLocationDialog._build_bbox(10.0, 20.0, 30.0, 40.0)
    assert bbox == GpsBoundingBox(south=10.0, west=20.0, north=30.0, east=40.0)


def test_build_bbox_invalid():
    assert MapLocationDialog._build_bbox(None, None, None, None) is None  # type: ignore[arg-type]


def test_map_dialog_selected_bbox_initially_none(qtbot, tmp_app_paths):
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)
    assert dialog.selected_bbox() is None


def test_map_dialog_extract_bbox(qtbot, tmp_app_paths):
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    root_obj = dialog._quick_widget.rootObject()
    if root_obj is None:
        pytest.skip("QML/QtLocation not available in this environment")
    else:
        root_obj.setProperty("south", 10.0)
        root_obj.setProperty("west", 20.0)
        root_obj.setProperty("north", 30.0)
        root_obj.setProperty("east", 40.0)

        bbox = dialog._extract_bbox()
        assert bbox == GpsBoundingBox(south=10.0, west=20.0, north=30.0, east=40.0)


# --- _on_qml_status_changed ---


def test_on_qml_status_changed_logs_errors_when_error(qtbot, caplog, tmp_app_paths):
    """_on_qml_status_changed logs every error message when status is Error."""
    from PySide6 import QtQuickWidgets

    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    dialog._quick_widget = MagicMock()
    dialog._quick_widget.errors.return_value = ["bad qml syntax"]

    with caplog.at_level(logging.ERROR, logger="photoaident.ui.widgets.map_dialog"):
        dialog._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Error)

    assert len(caplog.records) == 1
    assert "QML error" in caplog.records[0].message


def test_on_qml_status_changed_does_not_log_errors_when_ready(qtbot, tmp_app_paths):
    """_on_qml_status_changed never calls errors() when status is Ready."""
    from PySide6 import QtQuickWidgets

    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    dialog._quick_widget = MagicMock()
    dialog._quick_widget.rootObject.return_value = None

    dialog._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    dialog._quick_widget.errors.assert_not_called()


# --- _apply_initial_bbox ---


def _setproperty_calls(mock_root: MagicMock) -> dict:
    """Return a dict of {name: value} from all setProperty calls on mock_root."""
    return {call.args[0]: call.args[1] for call in mock_root.setProperty.call_args_list}


def test_apply_initial_bbox_pre_populates_extraction_properties(qtbot, tmp_app_paths):
    """_apply_initial_bbox sets south/west/north/east directly so _extract_bbox
    returns the correct value even before updateBbox() has run."""
    dialog = MapLocationDialog(paths=tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    bbox = GpsBoundingBox(south=48.0, north=52.0, west=10.0, east=14.0)
    dialog._apply_initial_bbox(mock_root, bbox)

    props = _setproperty_calls(mock_root)
    assert props["south"] == 48.0
    assert props["west"] == 10.0
    assert props["north"] == 52.0
    assert props["east"] == 14.0


def test_apply_initial_bbox_sets_pending_bbox_trigger(qtbot, tmp_app_paths):
    """_apply_initial_bbox sets pendingBbox=True last to trigger the QML view fit."""
    dialog = MapLocationDialog(paths=tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    bbox = GpsBoundingBox(south=48.0, north=52.0, west=10.0, east=14.0)
    dialog._apply_initial_bbox(mock_root, bbox)

    props = _setproperty_calls(mock_root)
    assert props["pendingBbox"] is True
    assert props["pendingBboxSouth"] == 48.0
    assert props["pendingBboxWest"] == 10.0
    assert props["pendingBboxNorth"] == 52.0
    assert props["pendingBboxEast"] == 14.0
    # pendingBbox must be set AFTER the coordinate properties so QML reads them
    last_call = mock_root.setProperty.call_args_list[-1]
    assert last_call.args[0] == "pendingBbox"


def test_apply_initial_bbox_antimeridian_passes_raw_coords():
    """_apply_initial_bbox passes antimeridian bbox coords to QML via setProperty."""
    root = MagicMock()
    bbox = GpsBoundingBox(south=30.0, west=170.0, north=50.0, east=-170.0)
    MapLocationDialog._apply_initial_bbox(root, bbox)
    props = _setproperty_calls(root)
    assert props["south"] == 30.0
    assert props["west"] == 170.0
    assert props["north"] == 50.0
    assert props["east"] == -170.0


def test_apply_initial_bbox_pacific_antimeridian_passes_raw_coords():
    """_apply_initial_bbox passes Pacific antimeridian coords to QML via setProperty."""
    root = MagicMock()
    bbox = GpsBoundingBox(south=20.0, west=160.0, north=40.0, east=-150.0)
    MapLocationDialog._apply_initial_bbox(root, bbox)
    props = _setproperty_calls(root)
    assert props["south"] == 20.0
    assert props["west"] == 160.0
    assert props["north"] == 40.0
    assert props["east"] == -150.0


# --- _extract_bbox ---


def test_extract_bbox_returns_none_when_no_root_object(qtbot, tmp_app_paths):
    """_extract_bbox returns None when rootObject() is None."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    dialog._quick_widget = MagicMock()
    dialog._quick_widget.rootObject.return_value = None

    assert dialog._extract_bbox() is None


# --- _on_accept ---


def test_on_accept_stores_selected_bbox(qtbot, tmp_app_paths):
    """_on_accept stores the bbox returned by _extract_bbox."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    expected = GpsBoundingBox(south=1.0, west=2.0, north=3.0, east=4.0)

    with (
        patch.object(dialog, "_extract_bbox", return_value=expected),
        patch.object(MapLocationDialog, "accept"),
    ):
        dialog._on_accept()

    assert dialog.selected_bbox() == expected


# --- zoom buttons ---


def test_zoom_buttons_exist(qtbot, tmp_app_paths):
    """Dialog has zoom-in and zoom-out buttons."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    assert hasattr(dialog, "_zoom_in_btn")
    assert hasattr(dialog, "_zoom_out_btn")
    assert dialog._zoom_in_btn.text() == "Zoom in"
    assert dialog._zoom_out_btn.text() == "Zoom out"


def test_on_zoom_in_calls_qml_zoom_in(qtbot, tmp_app_paths):
    """_on_zoom_in calls zoomIn() on the QML root object."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    dialog._quick_widget = MagicMock()
    dialog._quick_widget.rootObject.return_value = mock_root

    dialog._on_zoom_in()

    mock_root.zoomIn.assert_called_once_with()


def test_on_zoom_out_calls_qml_zoom_out(qtbot, tmp_app_paths):
    """_on_zoom_out calls zoomOut() on the QML root object."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    dialog._quick_widget = MagicMock()
    dialog._quick_widget.rootObject.return_value = mock_root

    dialog._on_zoom_out()

    mock_root.zoomOut.assert_called_once_with()


def test_on_zoom_in_noop_when_no_root_object(qtbot, tmp_app_paths):
    """_on_zoom_in is a no-op when rootObject() returns None."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    dialog._quick_widget = MagicMock()
    dialog._quick_widget.rootObject.return_value = None

    dialog._on_zoom_in()

    mock_root.zoomIn.assert_not_called()


def test_on_zoom_out_noop_when_no_root_object(qtbot, tmp_app_paths):
    """_on_zoom_out is a no-op when rootObject() returns None."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    dialog._quick_widget = MagicMock()
    dialog._quick_widget.rootObject.return_value = None

    dialog._on_zoom_out()

    mock_root.zoomOut.assert_not_called()
