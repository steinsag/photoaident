from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from photoaident.core.embeddings import FaceEmbedder
from photoaident.core.providers import select_providers
from photoaident.db.vector_store import VectorStore


def test_select_providers_cuda_first():
    with patch(
        "photoaident.core.providers.onnxruntime.get_available_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        providers = select_providers()
    assert providers[0] == "CUDAExecutionProvider"
    assert "CPUExecutionProvider" in providers


def test_select_providers_coreml_fallback():
    with patch(
        "photoaident.core.providers.onnxruntime.get_available_providers",
        return_value=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    ):
        providers = select_providers()
    assert providers[0] == "CoreMLExecutionProvider"
    assert "CPUExecutionProvider" in providers


def test_select_providers_cpu_only():
    with patch(
        "photoaident.core.providers.onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
        providers = select_providers()
    assert providers == ["CPUExecutionProvider"]


def test_select_providers_excludes_unavailable():
    with patch(
        "photoaident.core.providers.onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
        providers = select_providers()
    assert "CUDAExecutionProvider" not in providers
    assert "CoreMLExecutionProvider" not in providers


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


def test_process_image_raises_for_unreadable(tmp_path, face_embedder):
    """process_image raises UnidentifiedImageError for corrupt/unreadable files.

    The indexer's broad Exception handler logs and skips such files.
    """
    from PIL import UnidentifiedImageError

    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"not an image")

    with pytest.raises(UnidentifiedImageError):
        face_embedder.process_image(bad_file)


_rng = np.random.default_rng(seed=42)


def _make_mock_face() -> MagicMock:
    face = MagicMock()
    face.bbox = np.array([10.0, 20.0, 50.0, 60.0])
    face.normed_embedding = _rng.random(VectorStore.DEFAULT_DIMENSION).astype(
        VectorStore.EMBEDDING_DTYPE
    )
    face.det_score = VectorStore.EMBEDDING_DTYPE(0.98)
    face.gender = 1
    face.age = 25
    return face


def test_process_image_returns_face_data(tmp_path, face_embedder):
    """process_image returns the expected dict for each detected face."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    mock_face = _make_mock_face()

    with patch.object(face_embedder.app, "get", return_value=[mock_face]):
        result = face_embedder.process_image(img_path)

    assert len(result) == 1
    r = result[0]
    assert r["bbox"] == [10, 20, 50, 60]
    assert r["det_score"] == pytest.approx(0.98, abs=1e-4)
    assert r["gender"] == 1
    assert r["age"] == 25
    assert r["embedding"] is mock_face.normed_embedding


def test_process_image_returns_empty_when_no_faces(tmp_path, face_embedder):
    """process_image returns [] when the model detects no faces."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    with patch.object(face_embedder.app, "get", return_value=[]):
        result = face_embedder.process_image(img_path)

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


# ===========================================================================
# _prescale_for_detection
# ===========================================================================


def test_prescale_downscales_large_image():
    """A 4608×3456 image is scaled so its longest side equals max_dim=1920."""
    img = np.zeros((3456, 4608, 3), dtype=np.uint8)
    scaled, scale = FaceEmbedder._prescale_for_detection(img, max_dim=1920)

    assert max(scaled.shape[:2]) == 1920
    assert scale == pytest.approx(1920 / 4608, rel=1e-6)
    # Aspect ratio preserved
    assert scaled.shape[1] / scaled.shape[0] == pytest.approx(4608 / 3456, rel=0.01)


def test_prescale_no_op_for_small_image():
    """An image smaller than max_dim is returned unchanged (same object)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    scaled, scale = FaceEmbedder._prescale_for_detection(img, max_dim=1920)

    assert scaled is img
    assert scale == 1.0


def test_prescale_no_op_at_threshold():
    """An image with longest side exactly equal to max_dim is not rescaled."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    scaled, scale = FaceEmbedder._prescale_for_detection(img, max_dim=1920)

    assert scaled is img
    assert scale == 1.0


def test_process_image_scales_bbox_to_original_coords(tmp_path, face_embedder):
    """Bboxes returned by process_image are in original-image coordinates.

    When the image exceeds _MAX_DETECTION_DIM the detector sees a prescaled
    copy.  The bbox it returns must be mapped back to the original-image space
    by dividing by the scale factor.
    """
    # Image wider than _MAX_DETECTION_DIM so prescaling kicks in
    max_dim = FaceEmbedder._MAX_DETECTION_DIM
    img_w, img_h = max_dim + 480, max_dim - 100  # e.g. 2400 × 1820
    img_path = tmp_path / "large.jpg"
    Image.new("RGB", (img_w, img_h)).save(img_path)

    # Expected scale that _prescale_for_detection will apply
    expected_scale = max_dim / img_w

    # Bbox in prescaled coordinates that the mock detector will "return"
    scaled_bbox = np.array([100.0, 200.0, 300.0, 400.0])
    mock_face = _make_mock_face()
    mock_face.bbox = scaled_bbox

    with patch.object(face_embedder.app, "get", return_value=[mock_face]):
        result = face_embedder.process_image(img_path)

    assert len(result) == 1
    expected_bbox = (scaled_bbox / expected_scale).astype(int).tolist()
    assert result[0]["bbox"] == expected_bbox
