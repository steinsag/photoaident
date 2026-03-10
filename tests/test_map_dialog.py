import logging
import math
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


# --- _log_qml_errors ---


def test_log_qml_errors_when_status_error(qtbot, caplog, tmp_app_paths):
    """_log_qml_errors logs every error message when QML status is Error."""
    from PySide6 import QtQuickWidgets

    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    dialog._quick_widget = MagicMock()
    dialog._quick_widget.status.return_value = QtQuickWidgets.QQuickWidget.Status.Error
    dialog._quick_widget.errors.return_value = ["bad qml syntax"]

    with caplog.at_level(logging.ERROR, logger="photoaident.ui.widgets.map_dialog"):
        dialog._log_qml_errors()

    assert len(caplog.records) == 1
    assert "QML error" in caplog.records[0].message


def test_log_qml_errors_when_status_ok(qtbot, tmp_app_paths):
    """_log_qml_errors does nothing and never calls errors() when status is Ready."""
    from PySide6 import QtQuickWidgets

    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    dialog._quick_widget = MagicMock()
    dialog._quick_widget.status.return_value = QtQuickWidgets.QQuickWidget.Status.Ready

    dialog._log_qml_errors()

    dialog._quick_widget.errors.assert_not_called()


# --- _apply_initial_bbox ---


def test_apply_initial_bbox_sets_centre_and_zoom(qtbot, tmp_app_paths):
    """_apply_initial_bbox derives lat/lon center and zoom from bbox."""
    dialog = MapLocationDialog(paths=tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    bbox = GpsBoundingBox(south=48.0, north=52.0, west=10.0, east=14.0)
    dialog._apply_initial_bbox(mock_root, bbox)

    expected_zoom = int(max(2, min(15, round(math.log2(252.0 / 4.0)))))
    mock_root.setProperty.assert_any_call("initialLat", 50.0)
    mock_root.setProperty.assert_any_call("initialLon", 12.0)
    mock_root.setProperty.assert_any_call("initialZoom", expected_zoom)


def test_apply_initial_bbox_zero_span_uses_default_zoom(qtbot, tmp_app_paths):
    """_apply_initial_bbox falls back to zoom=14 when the bbox has zero span."""
    dialog = MapLocationDialog(tmp_app_paths)
    qtbot.addWidget(dialog)

    mock_root = MagicMock()
    bbox = GpsBoundingBox(south=50.0, north=50.0, west=10.0, east=10.0)
    dialog._apply_initial_bbox(mock_root, bbox)

    mock_root.setProperty.assert_any_call("initialZoom", 14)


def _apply_bbox(south: float, west: float, north: float, east: float) -> dict:
    """Helper: call _apply_initial_bbox and return the properties dict."""
    root = MagicMock()
    props: dict = {}
    root.setProperty.side_effect = lambda k, v: props.__setitem__(k, v)
    bbox = GpsBoundingBox(south=south, west=west, north=north, east=east)
    MapLocationDialog._apply_initial_bbox(root, bbox)
    return props


def test_apply_initial_bbox_antimeridian_span_is_short_arc():
    """Bbox crossing antimeridian uses the short arc span, not 360 - short."""
    # west=170, east=-170: crossing span = 20°, NOT 340°
    props = _apply_bbox(south=30.0, west=170.0, north=50.0, east=-170.0)
    expected_zoom = int(max(2, min(15, round(math.log2(252.0 / 20.0)))))
    assert props["initialZoom"] == expected_zoom


def test_apply_initial_bbox_antimeridian_center_lon_normalized():
    """Center longitude is normalized to [-180, 180] for antimeridian-crossing bbox."""
    # west=170, east=-170: center = 180° → normalized to -180 or 180
    props = _apply_bbox(south=30.0, west=170.0, north=50.0, east=-170.0)
    center = props["initialLon"]
    assert center == pytest.approx(180.0) or center == pytest.approx(-180.0)


def test_apply_initial_bbox_antimeridian_center_lon_pacific():
    """Center longitude is correct for a typical Pacific antimeridian bbox."""
    # west=160, east=-150: span=50°, center = 160 + 25 = 185 → -175
    props = _apply_bbox(south=20.0, west=160.0, north=40.0, east=-150.0)
    assert props["initialLon"] == pytest.approx(-175.0)


def test_apply_initial_bbox_antimeridian_center_lat_unaffected():
    """Latitude center is unaffected by antimeridian crossing."""
    props = _apply_bbox(south=20.0, west=160.0, north=40.0, east=-150.0)
    assert props["initialLat"] == pytest.approx(30.0)


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
