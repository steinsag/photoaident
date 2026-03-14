import logging
from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtQuickWidgets, QtWidgets
from pathlib import Path

from photoaident.core.geo import GpsBoundingBox
from photoaident.ui.widgets.map_dialog import MapLocationDialog, _icon_path


@pytest.fixture
def map_dialog(qtbot, tmp_app_paths):
    """A MapLocationDialog with no initial bbox, registered with qtbot.

    QQuickWidget.setSource is patched to prevent the QML engine from starting,
    which would leak file descriptors across tests.
    """
    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)
    return dialog


def _setproperty_calls(mock_root: MagicMock) -> dict:
    """Return a dict of {name: value} from all setProperty calls on *mock_root*."""
    return {call.args[0]: call.args[1] for call in mock_root.setProperty.call_args_list}


def _mock_quick_widget(dialog: MapLocationDialog, root_obj: object = None) -> None:
    """Replace the dialog's QQuickWidget with a mock, optionally providing a root."""
    mock_widget = MagicMock()
    mock_widget.rootObject.return_value = root_obj
    dialog._quick_widget = mock_widget


# --- _icon_path ---


def test_icon_path_dev_mode():
    """Returns the repo-relative assets/icons path in dev mode."""
    result = _icon_path("zoom-in.svg")
    assert Path(result).parts[-3:] == ("assets", "icons", "zoom-in.svg")


def test_icon_path_pyinstaller_bundle(tmp_path):
    """Returns the _MEIPASS-based path when running in a PyInstaller bundle."""
    with patch("photoaident.ui.widgets.map_dialog.sys") as mock_sys:
        mock_sys._MEIPASS = str(tmp_path)
        result = _icon_path("zoom-in.svg")

    assert result == str(tmp_path / "assets" / "icons" / "zoom-in.svg")


# --- _build_bbox ---


def test_build_bbox_valid():
    """Returns a GpsBoundingBox from valid coordinate values."""
    bbox = MapLocationDialog._build_bbox(10.0, 20.0, 30.0, 40.0)
    assert bbox is not None
    assert bbox.south == pytest.approx(10.0)
    assert bbox.west == pytest.approx(20.0)
    assert bbox.north == pytest.approx(30.0)
    assert bbox.east == pytest.approx(40.0)


def test_build_bbox_invalid():
    """Returns None when coordinates are not convertible to float."""
    assert MapLocationDialog._build_bbox(None, None, None, None) is None  # type: ignore[arg-type]


# --- selected_bbox ---


def test_selected_bbox_initially_none(map_dialog):
    """selected_bbox() returns None before the dialog is accepted."""
    assert map_dialog.selected_bbox() is None


# --- _extract_bbox ---


def test_extract_bbox_returns_none_when_no_root_object(map_dialog):
    """Returns None when rootObject() is None."""
    _mock_quick_widget(map_dialog, root_obj=None)
    assert map_dialog._extract_bbox() is None


# --- _on_qml_status_changed ---


def test_on_qml_status_changed_logs_errors(map_dialog, caplog):
    """Logs every QML error when status is Error."""
    _mock_quick_widget(map_dialog)
    map_dialog._quick_widget.errors.return_value = ["bad qml syntax"]

    with caplog.at_level(logging.ERROR, logger="photoaident.ui.widgets.map_dialog"):
        map_dialog._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Error)

    assert len(caplog.records) == 1
    assert "QML error" in caplog.records[0].message


def test_on_qml_status_changed_ready_without_root_does_not_call_errors(map_dialog):
    """Does not call errors() when status is Ready (even if rootObject is None)."""
    _mock_quick_widget(map_dialog, root_obj=None)

    map_dialog._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    map_dialog._quick_widget.errors.assert_not_called()


def test_on_qml_status_changed_ready_applies_initial_bbox(qtbot, tmp_app_paths):
    """Applies the initial bbox when QML becomes Ready and a bbox is pending."""
    initial = GpsBoundingBox(south=48.0, west=10.0, north=52.0, east=14.0)
    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        dialog = MapLocationDialog(tmp_app_paths, initial_bbox=initial)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    _mock_quick_widget(dialog, root_obj=mock_root)

    dialog._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    props = _setproperty_calls(mock_root)
    assert props["pendingBbox"] is True
    assert props["south"] == pytest.approx(48.0)
    assert props["north"] == pytest.approx(52.0)


# --- _apply_initial_bbox ---


BBOX_GERMANY = GpsBoundingBox(south=48.0, west=10.0, north=52.0, east=14.0)


def test_apply_initial_bbox_pre_populates_extraction_properties():
    """Sets south/west/north/east so _extract_bbox works before updateBbox()."""
    mock_root = MagicMock()
    MapLocationDialog._apply_initial_bbox(mock_root, BBOX_GERMANY)

    props = _setproperty_calls(mock_root)
    assert props["south"] == pytest.approx(48.0)
    assert props["west"] == pytest.approx(10.0)
    assert props["north"] == pytest.approx(52.0)
    assert props["east"] == pytest.approx(14.0)


def test_apply_initial_bbox_sets_pending_bbox_trigger_last():
    """Sets pendingBbox=True as the final property to trigger QML view fit."""
    mock_root = MagicMock()
    MapLocationDialog._apply_initial_bbox(mock_root, BBOX_GERMANY)

    props = _setproperty_calls(mock_root)
    assert props["pendingBbox"] is True
    assert props["pendingBboxSouth"] == pytest.approx(48.0)
    assert props["pendingBboxWest"] == pytest.approx(10.0)
    assert props["pendingBboxNorth"] == pytest.approx(52.0)
    assert props["pendingBboxEast"] == pytest.approx(14.0)
    last_call = mock_root.setProperty.call_args_list[-1]
    assert last_call.args[0] == "pendingBbox"


@pytest.mark.parametrize(
    "bbox",
    [
        GpsBoundingBox(south=30.0, west=170.0, north=50.0, east=-170.0),
        GpsBoundingBox(south=20.0, west=160.0, north=40.0, east=-150.0),
    ],
    ids=["dateline-170", "pacific-160"],
)
def test_apply_initial_bbox_antimeridian_passes_raw_coords(bbox):
    """Passes antimeridian bbox coordinates unchanged to QML."""
    mock_root = MagicMock()
    MapLocationDialog._apply_initial_bbox(mock_root, bbox)

    props = _setproperty_calls(mock_root)
    assert props["south"] == pytest.approx(bbox.south)
    assert props["west"] == pytest.approx(bbox.west)
    assert props["north"] == pytest.approx(bbox.north)
    assert props["east"] == pytest.approx(bbox.east)


# --- _on_accept ---


def test_on_accept_stores_selected_bbox(map_dialog):
    """Stores the bbox returned by _extract_bbox on accept."""
    expected = GpsBoundingBox(south=1.0, west=2.0, north=3.0, east=4.0)

    with (
        patch.object(map_dialog, "_extract_bbox", return_value=expected),
        patch.object(MapLocationDialog, "accept"),
    ):
        map_dialog._on_accept()

    assert map_dialog.selected_bbox() == expected


# --- done (geometry save) ---


def test_done_saves_widget_geometry(map_dialog):
    """done() calls save_widget_geometry before closing."""
    with patch("photoaident.ui.widgets.map_dialog.save_widget_geometry") as mock_save:
        map_dialog.done(QtWidgets.QDialog.DialogCode.Accepted)

    mock_save.assert_called_once_with(map_dialog, map_dialog._paths.window_state_file)


# --- zoom buttons ---


def test_zoom_buttons_exist(map_dialog):
    """Dialog has labelled zoom-in and zoom-out buttons."""
    assert map_dialog._zoom_in_btn.text() == "Zoom in"
    assert map_dialog._zoom_out_btn.text() == "Zoom out"


@pytest.mark.parametrize(
    "method,expected_delta",
    [("_on_zoom_in", 1), ("_on_zoom_out", -1)],
    ids=["zoom-in", "zoom-out"],
)
def test_zoom_sets_pending_zoom_delta(map_dialog, method, expected_delta):
    """Zoom handler sets pendingZoomDelta on the QML root object."""
    mock_root = MagicMock()
    _mock_quick_widget(map_dialog, root_obj=mock_root)

    getattr(map_dialog, method)()

    mock_root.setProperty.assert_called_once_with("pendingZoomDelta", expected_delta)


@pytest.mark.parametrize(
    "method",
    ["_on_zoom_in", "_on_zoom_out"],
    ids=["zoom-in", "zoom-out"],
)
def test_zoom_noop_when_no_root_object(map_dialog, method):
    """Zoom handler is a no-op when rootObject() returns None."""
    _mock_quick_widget(map_dialog, root_obj=None)

    getattr(map_dialog, method)()  # should not raise
