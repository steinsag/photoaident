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
