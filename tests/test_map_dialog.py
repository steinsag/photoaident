import pytest

from photoaident.core.geo import GpsBoundingBox
from photoaident.ui.widgets.map_dialog import MapLocationDialog


def test_build_bbox_valid():
    bbox = MapLocationDialog._build_bbox(10.0, 20.0, 30.0, 40.0)
    assert bbox == GpsBoundingBox(south=10.0, west=20.0, north=30.0, east=40.0)


def test_build_bbox_invalid():
    assert MapLocationDialog._build_bbox(None, None, None, None) is None  # type: ignore[arg-type]


def test_map_dialog_selected_bbox_initially_none(qtbot):
    dialog = MapLocationDialog()
    qtbot.addWidget(dialog)
    assert dialog.selected_bbox() is None


def test_map_dialog_extract_bbox(qtbot):
    dialog = MapLocationDialog()
    qtbot.addWidget(dialog)

    root_obj = dialog._quick_widget.rootObject()
    if root_obj is None:
        pytest.skip("QML/QtLocation not available in this environment")

    root_obj.setProperty("south", 10.0)
    root_obj.setProperty("west", 20.0)
    root_obj.setProperty("north", 30.0)
    root_obj.setProperty("east", 40.0)

    bbox = dialog._extract_bbox()
    assert bbox == GpsBoundingBox(south=10.0, west=20.0, north=30.0, east=40.0)
