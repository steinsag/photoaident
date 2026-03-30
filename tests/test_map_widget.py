import logging
from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtQuickWidgets

from photoaident.core.geo import GpsBoundingBox
from photoaident.ui.widgets.map_widget import (
    MapWidget,
    _apply_initial_bbox,
    _build_bbox,
)


@pytest.fixture
def map_widget(qtbot, tmp_app_paths):
    """A MapWidget with show_overlay=True, QML engine prevented from starting."""
    with patch.object(QtQuickWidgets.QQuickWidget, "setSource"):
        widget = MapWidget(tmp_app_paths, show_overlay=True)
    qtbot.addWidget(widget)
    return widget


def _mock_root(widget: MapWidget) -> MagicMock:
    """Replace the internal QQuickWidget's rootObject with a mock."""
    mock_root = MagicMock()
    widget._quick_widget = MagicMock()
    widget._quick_widget.rootObject.return_value = mock_root
    return mock_root


def _setproperty_calls(mock_root: MagicMock) -> dict:
    return {c.args[0]: c.args[1] for c in mock_root.setProperty.call_args_list}


# --- Construction ---


def test_construction_show_overlay_true(qtbot, tmp_app_paths):
    """setInitialProperties includes showOverlay=True."""
    with (
        patch.object(QtQuickWidgets.QQuickWidget, "setSource"),
        patch.object(
            QtQuickWidgets.QQuickWidget,
            "setInitialProperties",
        ) as mock_props,
    ):
        widget = MapWidget(tmp_app_paths, show_overlay=True)
    qtbot.addWidget(widget)
    args = mock_props.call_args[0][0]
    assert args["showOverlay"] is True


def test_construction_show_overlay_false(qtbot, tmp_app_paths):
    """setInitialProperties includes showOverlay=False."""
    with (
        patch.object(QtQuickWidgets.QQuickWidget, "setSource"),
        patch.object(
            QtQuickWidgets.QQuickWidget,
            "setInitialProperties",
        ) as mock_props,
    ):
        widget = MapWidget(tmp_app_paths, show_overlay=False)
    qtbot.addWidget(widget)
    args = mock_props.call_args[0][0]
    assert args["showOverlay"] is False


# --- Zoom buttons ---


def test_zoom_buttons_are_icon_only(map_widget):
    """Zoom buttons have no text label — icon-only."""
    assert map_widget._zoom_in_btn.text() == ""
    assert map_widget._zoom_out_btn.text() == ""


def test_zoom_in_sets_pending_zoom_delta(map_widget):
    """_on_zoom_in sets pendingZoomDelta=1 on QML root."""
    mock_root = _mock_root(map_widget)
    map_widget._on_zoom_in()
    mock_root.setProperty.assert_called_once_with("pendingZoomDelta", 1)


def test_zoom_out_sets_pending_zoom_delta(map_widget):
    """_on_zoom_out sets pendingZoomDelta=-1 on QML root."""
    mock_root = _mock_root(map_widget)
    map_widget._on_zoom_out()
    mock_root.setProperty.assert_called_once_with("pendingZoomDelta", -1)


def test_zoom_noop_when_no_root(map_widget):
    """Zoom handlers are no-ops when rootObject() returns None."""
    map_widget._quick_widget = MagicMock()
    map_widget._quick_widget.rootObject.return_value = None
    map_widget._on_zoom_in()  # should not raise
    map_widget._on_zoom_out()  # should not raise


# --- set_center ---


def test_set_center_applies_immediately_when_ready(map_widget):
    """set_center sets initialLat/Lon/Zoom on root when QML is ready."""
    map_widget._ready = True
    mock_root = _mock_root(map_widget)

    map_widget.set_center(48.137, 11.576, 14)

    props = _setproperty_calls(mock_root)
    assert props["initialLat"] == pytest.approx(48.137)
    assert props["initialLon"] == pytest.approx(11.576)
    assert props["initialZoom"] == 14


def test_set_center_buffers_when_not_ready(map_widget):
    """set_center buffers the operation until QML is ready."""
    assert not map_widget._ready
    map_widget.set_center(48.0, 11.0, 12)
    assert len(map_widget._pending_ops) == 1


def test_set_center_applied_on_ready(map_widget):
    """Buffered set_center is applied when QML status becomes Ready."""
    map_widget.set_center(48.0, 11.0, 12)
    mock_root = _mock_root(map_widget)

    map_widget._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    props = _setproperty_calls(mock_root)
    assert props["initialLat"] == pytest.approx(48.0)
    assert props["initialZoom"] == 12


# --- set_marker ---


def test_set_marker_applies_immediately_when_ready(map_widget):
    """set_marker sets showMarker=True and coordinates on root."""
    map_widget._ready = True
    mock_root = _mock_root(map_widget)

    map_widget.set_marker(48.137, 11.576)

    props = _setproperty_calls(mock_root)
    assert props["showMarker"] is True
    assert props["markerLat"] == pytest.approx(48.137)
    assert props["markerLon"] == pytest.approx(11.576)


def test_set_marker_buffers_when_not_ready(map_widget):
    """set_marker buffers the operation until QML is ready."""
    map_widget.set_marker(1.0, 2.0)
    assert len(map_widget._pending_ops) == 1


def test_set_marker_applied_on_ready(map_widget):
    """Buffered set_marker is applied when QML becomes Ready."""
    map_widget.set_marker(51.5, -0.1)
    mock_root = _mock_root(map_widget)

    map_widget._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    props = _setproperty_calls(mock_root)
    assert props["showMarker"] is True
    assert props["markerLat"] == pytest.approx(51.5)
    assert props["markerLon"] == pytest.approx(-0.1)


# --- set_initial_bbox ---

BBOX_GERMANY = GpsBoundingBox(south=48.0, west=10.0, north=52.0, east=14.0)


def test_set_initial_bbox_applies_immediately_when_ready(map_widget):
    """set_initial_bbox calls _apply_initial_bbox on root when ready."""
    map_widget._ready = True
    mock_root = _mock_root(map_widget)

    map_widget.set_initial_bbox(BBOX_GERMANY)

    props = _setproperty_calls(mock_root)
    assert props["south"] == pytest.approx(48.0)
    assert props["north"] == pytest.approx(52.0)
    assert props["pendingBbox"] is True


def test_set_initial_bbox_buffers_when_not_ready(map_widget):
    """set_initial_bbox buffers the operation until QML is ready."""
    map_widget.set_initial_bbox(BBOX_GERMANY)
    assert len(map_widget._pending_ops) == 1


def test_set_initial_bbox_applied_on_ready(map_widget):
    """Buffered set_initial_bbox is applied when QML becomes Ready."""
    map_widget.set_initial_bbox(BBOX_GERMANY)
    mock_root = _mock_root(map_widget)

    map_widget._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    props = _setproperty_calls(mock_root)
    assert props["pendingBbox"] is True
    assert props["pendingBboxSouth"] == pytest.approx(48.0)


# --- current_bbox ---


def test_current_bbox_reads_from_root(map_widget):
    """current_bbox() reads south/west/north/east from QML root."""
    mock_root = _mock_root(map_widget)
    mock_root.property.side_effect = lambda name: {
        "south": 48.0,
        "west": 10.0,
        "north": 52.0,
        "east": 14.0,
    }[name]

    bbox = map_widget.current_bbox()

    assert bbox is not None
    assert bbox.south == pytest.approx(48.0)
    assert bbox.west == pytest.approx(10.0)
    assert bbox.north == pytest.approx(52.0)
    assert bbox.east == pytest.approx(14.0)


def test_current_bbox_returns_none_when_no_root(map_widget):
    """current_bbox() returns None when rootObject() is None."""
    map_widget._quick_widget = MagicMock()
    map_widget._quick_widget.rootObject.return_value = None
    assert map_widget.current_bbox() is None


# --- QML status: Error ---


def test_on_qml_status_error_logs_errors(map_widget, caplog):
    """QML errors are logged when status is Error."""
    map_widget._quick_widget = MagicMock()
    map_widget._quick_widget.errors.return_value = ["syntax error"]

    with caplog.at_level(logging.ERROR, logger="photoaident.ui.widgets.map_widget"):
        map_widget._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Error)

    assert any("QML error" in r.message for r in caplog.records)


def test_on_qml_status_ready_clears_pending_ops(map_widget):
    """Pending ops list is empty after Ready fires."""
    map_widget.set_center(1.0, 2.0, 5)
    assert len(map_widget._pending_ops) == 1

    _mock_root(map_widget)
    map_widget._on_qml_status_changed(QtQuickWidgets.QQuickWidget.Status.Ready)

    assert map_widget._pending_ops == []


# --- _build_bbox ---


def test_build_bbox_valid():
    """Returns a GpsBoundingBox from valid float-convertible values."""
    bbox = _build_bbox(10.0, 20.0, 30.0, 40.0)
    assert bbox is not None
    assert bbox.south == pytest.approx(10.0)
    assert bbox.east == pytest.approx(40.0)


def test_build_bbox_invalid():
    """Returns None when values are not float-convertible."""
    assert _build_bbox(None, None, None, None) is None


# --- _apply_initial_bbox ---


def test_apply_initial_bbox_sets_extraction_and_pending_props():
    """Sets south/west/north/east AND pendingBbox* properties."""
    mock_root = MagicMock()
    _apply_initial_bbox(mock_root, BBOX_GERMANY)

    props = _setproperty_calls(mock_root)
    assert props["south"] == pytest.approx(48.0)
    assert props["pendingBboxNorth"] == pytest.approx(52.0)
    assert props["pendingBbox"] is True


def test_apply_initial_bbox_sets_pending_bbox_last():
    """pendingBbox=True is set last to trigger the QML view-fit handler."""
    mock_root = MagicMock()
    _apply_initial_bbox(mock_root, BBOX_GERMANY)

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
    """Antimeridian bounding box coordinates are passed unchanged to QML."""
    mock_root = MagicMock()
    _apply_initial_bbox(mock_root, bbox)

    props = _setproperty_calls(mock_root)
    assert props["south"] == pytest.approx(bbox.south)
    assert props["east"] == pytest.approx(bbox.east)
