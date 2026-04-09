"""Tests for photoaident.utils.image_utils."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from photoaident.utils.image_utils import (
    extract_and_save_face_crop,
    generate_thumbnail,
    open_image,
)

# EXIF tag number for Orientation
_EXIF_ORIENTATION_TAG = 274

# ===========================================================================
# Helpers
# ===========================================================================


def _save_rgb_jpeg(path: Path, size: tuple[int, int] = (10, 10)) -> Path:
    """Save a tiny solid-red RGB JPEG and return its path."""
    PILImage.new("RGB", size, color=(255, 0, 0)).save(path, "JPEG")
    return path


def _save_rgba_png(path: Path, size: tuple[int, int] = (10, 10)) -> Path:
    """Save a tiny RGBA PNG and return its path."""
    PILImage.new("RGBA", size, color=(0, 128, 255, 200)).save(path, "PNG")
    return path


# ===========================================================================
# open_image
# ===========================================================================


def test_open_image_happy_path_returns_image_with_correct_size(tmp_path):
    """open_image on a real JPEG returns an Image with the expected dimensions."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg", size=(10, 10))
    img = open_image(src)
    try:
        assert img.size == (10, 10)
    finally:
        img.close()


def test_open_image_happy_path_returns_pil_image_instance(tmp_path):
    """open_image returns a PIL Image object."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")
    img = open_image(src)
    try:
        assert isinstance(img, PILImage.Image)
    finally:
        img.close()


def test_open_image_exif_transpose_raises_closes_original_before_propagating(tmp_path):
    """open_image closes the original image when exif_transpose raises."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")

    mock_original = MagicMock(spec=PILImage.Image)
    mock_original.__enter__ = lambda s: s
    mock_original.__exit__ = MagicMock(return_value=False)

    with (
        patch("photoaident.utils.image_utils.Image.open", return_value=mock_original),
        patch(
            "photoaident.utils.image_utils.ImageOps.exif_transpose",
            side_effect=ValueError("bad EXIF"),
        ),
        pytest.raises(ValueError, match="bad EXIF"),
    ):
        open_image(src)

    mock_original.close.assert_called_once()


def test_open_image_exif_transpose_returns_different_object_closes_original(tmp_path):
    """open_image closes the original when exif_transpose returns a new image."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")

    mock_original = MagicMock(spec=PILImage.Image)
    rotated = PILImage.new("RGB", (10, 10), color=(0, 255, 0))

    with (
        patch("photoaident.utils.image_utils.Image.open", return_value=mock_original),
        patch(
            "photoaident.utils.image_utils.ImageOps.exif_transpose",
            return_value=rotated,
        ),
    ):
        result = open_image(src)

    try:
        mock_original.close.assert_called_once()
        assert result is rotated
    finally:
        rotated.close()


def test_open_image_exif_transpose_returns_same_object_does_not_close_original(
    tmp_path,
):
    """open_image does not call close on the original when no rotation occurs."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")

    mock_original = MagicMock(spec=PILImage.Image)

    with (
        patch("photoaident.utils.image_utils.Image.open", return_value=mock_original),
        patch(
            "photoaident.utils.image_utils.ImageOps.exif_transpose",
            side_effect=lambda img: img,
        ),
    ):
        result = open_image(src)

    assert result is mock_original
    mock_original.close.assert_not_called()


def test_open_image_applies_exif_rotation(tmp_path):
    """open_image applies EXIF orientation so the returned image has visual dimensions.

    A 80×160 JPEG saved with orientation 6 (90° CW) should be returned as
    160×80 after exif_transpose corrects it.  This catches regressions if
    open_image() is modified to skip exif_transpose().
    """
    src = tmp_path / "rotated.jpg"

    # Create a portrait image (80 wide, 160 tall) and embed orientation=6 (90° CW).
    # Viewers and PIL would display it as landscape (160 wide, 80 tall).
    img = PILImage.new("RGB", (80, 160), color=(255, 0, 0))
    exif = img.getexif()
    exif[_EXIF_ORIENTATION_TAG] = 6
    img.save(src, "JPEG", exif=exif.tobytes())

    with open_image(src) as result:
        assert result.size == (
            160,
            80,
        ), f"Expected (160, 80) after EXIF transpose, got {result.size}"


# ===========================================================================
# generate_thumbnail
# ===========================================================================


def test_generate_thumbnail_creates_output_file(tmp_path):
    """generate_thumbnail saves a file at the given output path."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")
    out = tmp_path / "thumb.jpg"

    generate_thumbnail(src, out)

    assert out.exists()


def test_generate_thumbnail_output_is_jpeg(tmp_path):
    """generate_thumbnail writes a valid JPEG file."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")
    out = tmp_path / "thumb.jpg"

    generate_thumbnail(src, out)

    with PILImage.open(out) as img:
        assert img.format == "JPEG"


def test_generate_thumbnail_size_within_requested_bounds(tmp_path):
    """generate_thumbnail produces an image no larger than the requested size."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg", size=(100, 80))
    out = tmp_path / "thumb.jpg"
    max_size = (40, 40)

    generate_thumbnail(src, out, size=max_size)

    with PILImage.open(out) as img:
        assert img.width <= max_size[0]
        assert img.height <= max_size[1]


def test_generate_thumbnail_creates_intermediate_directories(tmp_path):
    """generate_thumbnail creates missing parent directories automatically."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg")
    out = tmp_path / "a" / "b" / "c" / "thumb.jpg"

    assert not out.parent.exists()

    generate_thumbnail(src, out)

    assert out.exists()


def test_generate_thumbnail_rgba_source_converts_to_rgb(tmp_path):
    """generate_thumbnail converts an RGBA source to RGB before saving as JPEG."""
    src = _save_rgba_png(tmp_path / "source.png")
    out = tmp_path / "thumb.jpg"

    generate_thumbnail(src, out)

    with PILImage.open(out) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


def test_generate_thumbnail_default_size_is_200x200(tmp_path):
    """generate_thumbnail uses (200, 200) as the default maximum size."""
    src = _save_rgb_jpeg(tmp_path / "source.jpg", size=(400, 300))
    out = tmp_path / "thumb.jpg"

    generate_thumbnail(src, out)

    with PILImage.open(out) as img:
        assert img.width <= 200
        assert img.height <= 200


def test_generate_thumbnail_preserves_aspect_ratio(tmp_path):
    """generate_thumbnail maintains the source aspect ratio."""
    # 200×100 => 2:1 aspect; thumbnail capped at (100, 100) → 100×50
    src = _save_rgb_jpeg(tmp_path / "source.jpg", size=(200, 100))
    out = tmp_path / "thumb.jpg"

    generate_thumbnail(src, out, size=(100, 100))

    with PILImage.open(out) as img:
        ratio_src = 200 / 100
        ratio_thumb = img.width / img.height
        assert abs(ratio_thumb - ratio_src) < 0.1


# ===========================================================================
# extract_and_save_face_crop
# ===========================================================================


def _make_pil_image(size: tuple[int, int] = (300, 300)) -> PILImage.Image:
    """Return a small solid-red PIL Image."""
    return PILImage.new("RGB", size, color=(255, 0, 0))


def test_extract_and_save_face_crop_success_returns_true(tmp_path):
    """extract_and_save_face_crop returns True when FaceEmbedder succeeds."""
    output_path = tmp_path / "out" / "crop.jpg"
    pil_image = _make_pil_image()

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        return_value=pil_image,
    ):
        result = extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(10, 20, 50, 60),
            output_path=output_path,
        )

    assert result is True


def test_extract_and_save_face_crop_success_creates_jpeg(tmp_path):
    """extract_and_save_face_crop writes a valid JPEG file at output_path."""
    output_path = tmp_path / "faces" / "crop.jpg"
    pil_image = _make_pil_image()

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        return_value=pil_image,
    ):
        extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(10, 20, 50, 60),
            output_path=output_path,
        )

    assert output_path.exists()
    with PILImage.open(output_path) as img:
        assert img.format == "JPEG"


def test_extract_and_save_face_crop_creates_intermediate_directories(tmp_path):
    """extract_and_save_face_crop creates missing parent directories."""
    output_path = tmp_path / "a" / "b" / "c" / "crop.jpg"
    assert not output_path.parent.exists()

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        return_value=_make_pil_image(),
    ):
        extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(0, 0, 100, 100),
            output_path=output_path,
        )

    assert output_path.exists()


def test_extract_and_save_face_crop_bbox_conversion(tmp_path):
    """extract_and_save_face_crop converts (x, y, w, h) to [x, y, x+w, y+h]."""
    output_path = tmp_path / "crop.jpg"
    mock_extract = MagicMock(return_value=_make_pil_image())

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        mock_extract,
    ):
        extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(10, 20, 50, 60),
            output_path=output_path,
        )

    _, xyxy_arg, _ = mock_extract.call_args.args
    assert xyxy_arg == [10, 20, 60, 80]


def test_extract_and_save_face_crop_passes_target_size(tmp_path):
    """extract_and_save_face_crop forwards target_size to FaceEmbedder."""
    output_path = tmp_path / "crop.jpg"
    mock_extract = MagicMock(return_value=_make_pil_image())

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        mock_extract,
    ):
        extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(0, 0, 50, 50),
            output_path=output_path,
            target_size=(128, 128),
        )

    _, _, size_arg = mock_extract.call_args.args
    assert size_arg == (128, 128)


def test_extract_and_save_face_crop_exception_returns_false(tmp_path):
    """extract_and_save_face_crop returns False when FaceEmbedder raises."""
    output_path = tmp_path / "crop.jpg"

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        side_effect=RuntimeError("ONNX exploded"),
    ):
        result = extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(0, 0, 50, 50),
            output_path=output_path,
        )

    assert result is False


def test_extract_and_save_face_crop_exception_no_file_created(tmp_path):
    """extract_and_save_face_crop does not create output_path on failure."""
    output_path = tmp_path / "crop.jpg"

    with patch(
        "photoaident.core.embeddings.FaceEmbedder.extract_face_crop",
        side_effect=RuntimeError("ONNX exploded"),
    ):
        extract_and_save_face_crop(
            image_path=tmp_path / "photo.jpg",
            bbox=(0, 0, 50, 50),
            output_path=output_path,
        )

    assert not output_path.exists()
