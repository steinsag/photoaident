from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps
from PySide6 import QtGui


def get_exif_transform(
    transformation: QtGui.QImageIOHandler.Transformation,
) -> QtGui.QTransform:
    """Return the QTransform that corresponds to a QImageIOHandler.Transformation.

    Replicates the logic Qt applies internally when
    ``QImageReader.setAutoTransform(True)`` is used, so callers can apply the
    same orientation correction manually after drawing overlays in the
    un-rotated coordinate space.

    The implementation mirrors the switch statement in Qt's own
    ``exifTransform`` helper (qtbase/src/gui/image/qimagereader.cpp).
    """
    m = QtGui.QTransform()
    T = QtGui.QImageIOHandler.Transformation
    if transformation == T.TransformationMirror:
        m.scale(-1.0, 1.0)
    elif transformation == T.TransformationFlip:
        m.scale(1.0, -1.0)
    elif transformation == T.TransformationRotate180:
        m.rotate(180.0)
    elif transformation == T.TransformationRotate90:
        m.rotate(90.0)
    elif transformation == T.TransformationMirrorAndRotate90:
        m.scale(-1.0, 1.0)
        m.rotate(90.0)
    elif transformation == T.TransformationFlipAndRotate90:
        m.scale(1.0, -1.0)
        m.rotate(90.0)
    elif transformation == T.TransformationRotate270:
        m.rotate(270.0)
    # TransformationNone → identity, already initialised above
    return m


def open_image(image_path: Path) -> Image.Image:
    """Open an image and apply EXIF orientation.

    All PIL image loading should go through this function so that the
    pixel layout matches what OpenCV (and therefore InsightFace) sees.
    The caller is responsible for closing the returned image.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img


def generate_thumbnail(
    image_path: Path, output_path: Path, size: Tuple[int, int] = (200, 200)
) -> None:
    """Generate a thumbnail for an image and save it to output_path.

    The thumbnail is generated while preserving aspect ratio and
    respecting EXIF orientation.
    """
    with open_image(image_path) as img:
        # Convert to RGB if necessary (e.g. for RGBA or CMYK)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.thumbnail(size, Image.Resampling.LANCZOS)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JPEG
        img.save(output_path, "JPEG", quality=85)
