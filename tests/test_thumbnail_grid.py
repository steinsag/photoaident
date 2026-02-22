import pytest
from PIL import Image as PILImage
from PySide6 import QtCore, QtWidgets

from photoaident.db.database import Face
from photoaident.ui.widgets.thumbnail_grid import ThumbnailWidget, ThumbnailGrid


@pytest.fixture
def sample_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = PILImage.new("RGB", (800, 600), "blue")
    img.save(img_path, "JPEG")
    return img_path


def test_thumbnail_widget_init(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb.jpg"
    faces = []
    widget = ThumbnailWidget(
        image_id=1,
        file_path=str(sample_image),
        faces=faces,
        thumb_path=thumb_path,
        orig_size=(800, 600),
    )
    qtbot.addWidget(widget)

    assert widget.image_id == 1
    assert widget.file_path == str(sample_image)
    assert widget.faces == faces
    assert widget.thumb_path == thumb_path
    assert widget.orig_size == (800, 600)

    # Check if thumbnail was generated because it didn't exist
    assert thumb_path.exists()
    assert widget.image_label.pixmap() is not None
    assert not widget.image_label.pixmap().isNull()


def test_thumbnail_widget_with_existing_thumb(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb_exists.jpg"
    # Create a dummy thumbnail
    PILImage.new("RGB", (150, 150), "red").save(thumb_path, "JPEG")

    widget = ThumbnailWidget(
        image_id=1, file_path=str(sample_image), faces=[], thumb_path=thumb_path
    )
    qtbot.addWidget(widget)

    # It should use the existing thumbnail
    assert thumb_path.exists()
    # Check if it loaded (label should have pixmap)
    assert widget.image_label.pixmap() is not None


def test_thumbnail_widget_click(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb_click.jpg"
    widget = ThumbnailWidget(1, str(sample_image), [], thumb_path)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.clicked) as blocker:
        qtbot.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)

    assert blocker.args == [1]


def test_thumbnail_widget_with_faces(qtbot, sample_image, tmp_path):
    thumb_path = tmp_path / "thumb_faces.jpg"
    face = Face(bbox_x=100, bbox_y=100, bbox_w=200, bbox_h=200)
    widget = ThumbnailWidget(
        image_id=1,
        file_path=str(sample_image),
        faces=[face],
        thumb_path=thumb_path,
        orig_size=(800, 600),
    )
    qtbot.addWidget(widget)

    # Face drawing logic is inside _load_thumbnail.
    # Hard to verify pixels without heavy lifting, but we can ensure it ran.
    assert widget.image_label.pixmap() is not None


def test_thumbnail_widget_error_loading(qtbot, tmp_path):
    # Pass a non-existent file path and non-existent thumb path
    thumb_path = tmp_path / "non_existent_thumb.jpg"
    widget = ThumbnailWidget(1, "non_existent_file.jpg", [], thumb_path)
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
    grid.add_thumbnail(1, str(sample_image), [], thumb_path, (800, 600))

    assert len(grid.thumbnails) == 1
    # Check if widget was added to layout
    assert grid.grid_layout.count() == 1


def test_thumbnail_grid_clear(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    grid.add_thumbnail(1, str(sample_image), [], tmp_path / "t1.jpg")
    grid.add_thumbnail(2, str(sample_image), [], tmp_path / "t2.jpg")

    assert len(grid.thumbnails) == 2
    assert grid.grid_layout.count() == 2

    grid.clear()

    assert len(grid.thumbnails) == 0
    assert grid.grid_layout.count() == 0


def test_thumbnail_grid_set_images(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    images_data = [
        (1, str(sample_image), [], tmp_path / "t1.jpg", (800, 600)),
        (2, str(sample_image), [], tmp_path / "t2.jpg", (800, 600)),
    ]

    grid.set_images_with_total(images_data, 10)

    assert len(grid.thumbnails) == 2
    assert grid.grid_layout.count() == 2
    # Should have a label "Showing 2 of 10" in main_layout
    assert grid.main_layout.count() == 2  # scroll area + label

    item = grid.main_layout.itemAt(1)
    assert item is not None
    label = item.widget()
    assert isinstance(label, QtWidgets.QLabel)
    assert "Showing first 2 of 10 images." in label.text()


def test_thumbnail_grid_set_images_empty_with_total(qtbot):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)
    grid.set_images_with_total([], 5)
    assert grid.main_layout.count() == 2
    item = grid.main_layout.itemAt(1)
    assert item is not None
    label = item.widget()
    assert isinstance(label, QtWidgets.QLabel)
    assert label.text() == "No images found."


def test_thumbnail_grid_resize(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    grid.show()  # Need to show to have a valid width
    qtbot.addWidget(grid)

    # Force a width that should result in a specific number of columns
    # 170 * 2 + 20 = 360
    grid.resize(400, 600)
    assert grid.cols == 2

    grid.add_thumbnail(1, str(sample_image), [], tmp_path / "t1.jpg")
    grid.add_thumbnail(2, str(sample_image), [], tmp_path / "t2.jpg")
    grid.add_thumbnail(3, str(sample_image), [], tmp_path / "t3.jpg")

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


def test_thumbnail_grid_set_images_simple(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)
    images_data = [(1, str(sample_image), [], tmp_path / "t1.jpg", (800, 600))]
    grid.set_images(images_data)
    assert len(grid.thumbnails) == 1
    assert grid.main_layout.count() == 1


def test_thumbnail_grid_set_images_no_total(qtbot, sample_image, tmp_path):
    grid = ThumbnailGrid()
    qtbot.addWidget(grid)

    images_data = [
        (1, str(sample_image), [], tmp_path / "t1.jpg", (800, 600)),
    ]

    # Case where total == len(images_data), should not show "Showing X of Y" label
    grid.set_images_with_total(images_data, 1)

    assert grid.main_layout.count() == 1  # only scroll area
