import pytest
from PIL import Image as PILImage
from PySide6 import QtCore

from photoaident.ui.widgets.thumbnail_grid import (
    PAGE_SIZE,
    ThumbnailGrid,
    ThumbnailWidget,
)


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = PILImage.new("RGB", (800, 600), "blue")
    img.save(img_path, "JPEG")
    return img_path


def test_thumbnail_widget_init(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb.jpg"
    widget = ThumbnailWidget(
        image_id=1,
        file_path=str(sample_image),
        thumb_path=thumb_path,
    )
    qtbot.addWidget(widget)

    assert widget.image_id == 1
    assert widget.file_path == str(sample_image)
    assert widget.thumb_path == thumb_path

    # Check if thumbnail was generated because it didn't exist
    assert thumb_path.exists()
    assert widget.image_label.pixmap() is not None
    assert not widget.image_label.pixmap().isNull()


def test_thumbnail_widget_with_existing_thumb(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb_exists.jpg"
    # Create a dummy thumbnail
    PILImage.new("RGB", (150, 150), "red").save(thumb_path, "JPEG")

    widget = ThumbnailWidget(
        image_id=1, file_path=str(sample_image), thumb_path=thumb_path
    )
    qtbot.addWidget(widget)

    # It should use the existing thumbnail
    assert thumb_path.exists()
    # Check if it loaded (label should have pixmap)
    assert widget.image_label.pixmap() is not None


def test_thumbnail_widget_click(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb_click.jpg"
    widget = ThumbnailWidget(1, str(sample_image), thumb_path)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.clicked) as blocker:
        qtbot.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)

    assert blocker.args == [1]


def test_thumbnail_widget_error_loading(qtbot, tmp_path):
    # Pass a non-existent file path and non-existent thumb path
    thumb_path = tmp_path / "non_existent_thumb.jpg"
    widget = ThumbnailWidget(1, "non_existent_file.jpg", thumb_path)
    qtbot.addWidget(widget)

    # Should show error text
    assert widget.image_label.text() == "Error loading image"


def test_thumbnail_grid_init(qtbot):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)
    assert len(grid.thumbnails) == 0


def test_thumbnail_grid_add_thumbnail(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    thumb_path = tmp_path / "thumb_grid.jpg"
    grid.add_thumbnail(1, str(sample_image), thumb_path)

    assert len(grid.thumbnails) == 1
    # Check if widget was added to layout
    assert grid.grid_layout.count() == 1


def test_thumbnail_grid_clear(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    grid.add_thumbnail(1, str(sample_image), tmp_path / "t1.jpg")
    grid.add_thumbnail(2, str(sample_image), tmp_path / "t2.jpg")

    assert len(grid.thumbnails) == 2
    assert grid.grid_layout.count() == 2

    grid.clear()

    assert len(grid.thumbnails) == 0
    assert grid.grid_layout.count() == 0


def test_thumbnail_grid_resize(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    grid.show()  # Need to show to have a valid width
    qtbot.addWidget(grid)

    # Force a width that should result in a specific number of columns
    # 170 * 2 + 20 = 360
    grid.resize(400, 600)
    assert grid.cols == 2

    grid.add_thumbnail(1, str(sample_image), tmp_path / "t1.jpg")
    grid.add_thumbnail(2, str(sample_image), tmp_path / "t2.jpg")
    grid.add_thumbnail(3, str(sample_image), tmp_path / "t3.jpg")

    # Check positions
    pos0 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[0])
    )
    assert isinstance(pos0, tuple)
    assert pos0[:2] == (0, 0)
    pos1 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[1])
    )
    assert isinstance(pos1, tuple)
    assert pos1[:2] == (0, 1)
    pos2 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[2])
    )
    assert isinstance(pos2, tuple)
    assert pos2[:2] == (1, 0)

    # Resize to 3 columns: 170 * 3 + 20 = 530
    grid.resize(600, 600)
    assert grid.cols == 3

    pos0 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[0])
    )
    assert isinstance(pos0, tuple)
    assert pos0[:2] == (0, 0)
    pos1 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[1])
    )
    assert isinstance(pos1, tuple)
    assert pos1[:2] == (0, 1)
    pos2 = grid.grid_layout.getItemPosition(
        grid.grid_layout.indexOf(grid.thumbnails[2])
    )
    assert isinstance(pos2, tuple)
    assert pos2[:2] == (0, 2)


# --- set_results / infinite scroll tests ---


def test_set_results_loads_first_page_only(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)]
    grid.set_results(results)

    # Only the first page is loaded synchronously; deferred fill not yet triggered
    assert len(grid.thumbnails) == PAGE_SIZE
    assert grid._loaded_count == PAGE_SIZE


def test_set_results_fewer_than_page_size(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(10)]
    grid.set_results(results)

    assert len(grid.thumbnails) == 10
    assert grid._hint_label.isHidden()


def test_set_results_empty(qtbot):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    grid.set_results([])

    assert len(grid.thumbnails) == 0
    assert grid._hint_label.isHidden()


def test_set_results_emits_results_changed(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(50)]

    received = []
    grid.results_changed.connect(lambda n: received.append(n))
    grid.set_results(results)

    assert received == [50]


def test_page_loaded_signal(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)]

    signals: list[tuple[int, int]] = []
    grid.page_loaded.connect(lambda loaded, total: signals.append((loaded, total)))
    grid.set_results(results)

    assert len(signals) >= 1
    assert signals[0] == (PAGE_SIZE, 100)


def test_set_results_resets_previous(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results1 = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)]
    grid.set_results(results1)
    assert len(grid.thumbnails) == PAGE_SIZE

    results2 = [(i, str(sample_image), tmp_path / f"s{i}.jpg") for i in range(100)]
    grid.set_results(results2)

    assert len(grid.thumbnails) == PAGE_SIZE
    assert grid._loaded_count == PAGE_SIZE


def test_hint_label_shown_when_more_remain(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)]
    grid.set_results(results)

    assert not grid._hint_label.isHidden()
    # 100 - PAGE_SIZE (30) = 70 remaining
    assert str(100 - PAGE_SIZE) in grid._hint_label.text()
    assert "remaining" in grid._hint_label.text()


def test_hint_label_hidden_when_all_loaded(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(10)]
    grid.set_results(results)

    assert grid._hint_label.isHidden()


def test_scroll_triggers_load_more(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    results = [(i, str(sample_image), tmp_path / f"t{i}.jpg") for i in range(100)]
    grid.set_results(results)

    initial_count = grid._loaded_count
    assert initial_count == PAGE_SIZE

    # Directly invoke scroll handler with a value that satisfies the condition
    grid._on_scroll_changed(grid.scroll_area.verticalScrollBar().maximum())

    assert grid._loaded_count == initial_count + PAGE_SIZE
