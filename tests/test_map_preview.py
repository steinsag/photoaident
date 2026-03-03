from photoaident.core.geo import GpsBoundingBox
from photoaident.ui.widgets.map_preview import MapPreviewWidget


def test_map_preview_initial_state(qtbot):
    widget = MapPreviewWidget()
    qtbot.addWidget(widget)

    assert widget._content_btn.text() == "Click to set location"
    assert widget._clear_btn.isHidden()


def test_map_preview_set_bbox(qtbot):
    widget = MapPreviewWidget()
    qtbot.addWidget(widget)

    bbox = GpsBoundingBox(south=10, west=20, north=30, east=40)
    widget.set_bbox(bbox)

    assert "Lat: 20.0000" in widget._content_btn.text()
    assert not widget._clear_btn.isHidden()


def test_map_preview_clear_bbox(qtbot):
    widget = MapPreviewWidget()
    qtbot.addWidget(widget)

    widget.set_bbox(GpsBoundingBox(0, 0, 1, 1))
    widget.set_bbox(None)

    assert widget._content_btn.text() == "Click to set location"
    assert widget._clear_btn.isHidden()
