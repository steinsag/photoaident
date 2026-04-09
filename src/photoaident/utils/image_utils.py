import logging
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def open_image(image_path: Path) -> Image.Image:
    """Open an image and apply EXIF orientation.

    All PIL image loading should go through this function so that the
    pixel layout matches what OpenCV (and therefore InsightFace) sees.
    The caller is responsible for closing the returned image.
    """
    original = Image.open(image_path)
    try:
        transposed = ImageOps.exif_transpose(original)
    except Exception:
        original.close()
        raise
    # exif_transpose returns a new Image when rotation is needed; close the
    # original in that case so we don't leak the file handle.
    if transposed is not original:
        original.close()
    return transposed


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


def extract_and_save_face_crop(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    output_path: Path,
    target_size: tuple[int, int] = (300, 300),
) -> bool:
    """Extract a face crop from an image and save it to output_path.

    The crop is written to a temporary file first and atomically replaced on
    success, so a failed or interrupted save never leaves a corrupt file that
    would be mistaken for a valid cache entry.

    Args:
        image_path: Path to the source image.
        bbox: (x, y, w, h) bounding box as stored in the database.
        output_path: Where to save the JPEG crop.
        target_size: Resize target for the saved crop.

    Returns:
        True on success, False if extraction fails.
    """
    import tempfile

    # Lazy import to avoid circular dependency and heavy ONNX load at module level.
    from photoaident.core.embeddings import FaceEmbedder

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        x, y, w, h = bbox
        xyxy = [x, y, x + w, y + h]
        crop = FaceEmbedder.extract_face_crop(image_path, xyxy, target_size)
        fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
        tmp_path = Path(tmp)
        try:
            import os

            os.close(fd)
            crop.save(tmp_path, "JPEG", quality=90)
            tmp_path.replace(output_path)
            tmp_path = None  # ownership transferred
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
        return True
    except Exception:
        logger.warning(
            "Failed to extract face crop for %s (bbox=%s, output=%s)",
            image_path,
            bbox,
            output_path,
            exc_info=True,
        )
        return False
