from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from photoaident.core.embeddings import FaceEmbedder


def test_face_embedder_init():
    # Test initialization (CPU for tests usually)
    embedder = FaceEmbedder(ctx_id=-1)
    assert embedder.app is not None


@pytest.mark.gpu
def test_face_embedder_gpu():
    embedder = FaceEmbedder(ctx_id=0)
    # Check if it actually uses CUDA if requested?
    # InsightFace doesn't make it easy to check without internal access.
    # We can at least check if it initialized.
    assert embedder.app is not None


def test_extract_face_crop(tmp_path):
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    bbox = [10, 10, 50, 50]
    crop = FaceEmbedder.extract_face_crop(img_path, bbox, target_size=(20, 20))

    assert crop.size == (20, 20)
    # Check if it's still red-ish
    pixel = crop.getpixel((5, 5))
    if isinstance(pixel, tuple):
        assert pixel[0] > 200  # Red component
    elif isinstance(pixel, int):
        assert pixel > 200


def test_process_image_returns_empty_when_unreadable(tmp_path):
    """process_image returns [] when cv2.imread cannot read the file."""
    embedder = FaceEmbedder(ctx_id=-1)
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"not an image")

    with patch("photoaident.core.embeddings.cv2.imread", return_value=None):
        result = embedder.process_image(bad_file)

    assert result == []


def _make_mock_face() -> MagicMock:
    face = MagicMock()
    face.bbox = np.array([10.0, 20.0, 50.0, 60.0])
    face.normed_embedding = np.random.rand(512).astype(np.float32)
    face.det_score = np.float32(0.98)
    face.gender = 1
    face.age = 25
    return face


def test_process_image_returns_face_data(tmp_path):
    """process_image returns the expected dict for each detected face."""
    embedder = FaceEmbedder(ctx_id=-1)
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    mock_face = _make_mock_face()

    with patch.object(embedder.app, "get", return_value=[mock_face]):
        result = embedder.process_image(img_path)

    assert len(result) == 1
    r = result[0]
    assert r["bbox"] == [10, 20, 50, 60]
    assert r["det_score"] == pytest.approx(0.98, abs=1e-4)
    assert r["gender"] == 1
    assert r["age"] == 25
    assert r["embedding"] is mock_face.normed_embedding


def test_process_image_returns_empty_when_no_faces(tmp_path):
    """process_image returns [] when the model detects no faces."""
    embedder = FaceEmbedder(ctx_id=-1)
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    with patch.object(embedder.app, "get", return_value=[]):
        result = embedder.process_image(img_path)

    assert result == []


def test_extract_face_crop_clamps_to_image_bounds(tmp_path):
    """A bbox that extends outside the image is clamped correctly."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), "green").save(img_path)

    # bbox with negative origin and far corner beyond image size
    bbox = [-10, -10, 150, 150]
    crop = FaceEmbedder.extract_face_crop(img_path, bbox, target_size=(50, 50))

    # Result must fit within target_size
    assert crop.size[0] <= 50
    assert crop.size[1] <= 50
    assert crop.mode == "RGB"
