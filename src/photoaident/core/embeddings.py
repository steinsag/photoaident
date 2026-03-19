from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from photoaident.core.providers import select_providers
from photoaident.utils.image_utils import open_image


class FaceEmbedder:
    """Wrapper for InsightFace to detect faces and compute embeddings."""

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        """Initialize the FaceAnalysis app.

        Args:
            model_name: The name of the InsightFace model bundle.
            ctx_id: Context ID passed to InsightFace (0 = first device, -1 = CPU).
        """
        providers = select_providers()
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def process_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detect faces and compute embeddings for an image.

        Args:
            image_path: Path to the image file.

        Returns:
            A list of dictionaries, each containing:
            - bbox: [x1, y1, x2, y2]
            - embedding: 512-dim vector
            - det_score: detection confidence
            - gender: 0 or 1
            - age: estimated age
        """
        # Load via PIL so EXIF orientation is applied before detection.
        # This ensures bboxes are in the same rotated coordinate space that
        # extract_face_crop() uses, preventing misaligned crops.
        with open_image(image_path) as pil_img:
            rgb_array = np.array(pil_img.convert("RGB"))

        if rgb_array.size == 0:
            return []

        # InsightFace expects BGR
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr_array)

        results = []
        for face in faces:
            results.append(
                {
                    "bbox": face.bbox.astype(int).tolist(),
                    "embedding": face.normed_embedding,
                    "det_score": float(face.det_score),
                    "gender": int(face.gender),
                    "age": int(face.age),
                }
            )

        return results

    @staticmethod
    def extract_face_crop(
        image_path: Path, bbox: List[int], target_size: Tuple[int, int] = (224, 224)
    ) -> Image.Image:
        """Extract and resize a face crop from an image.

        Args:
            image_path: Path to the original image.
            bbox: [x1, y1, x2, y2] bounding box.
            target_size: Size to resize the crop to.

        Returns:
            A PIL Image of the face crop.
        """
        with open_image(image_path) as img:

            # bbox is [x1, y1, x2, y2]
            # Ensure bbox is within image bounds
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)

            crop = img.crop((x1, y1, x2, y2))
            crop = crop.convert("RGB")
            crop.thumbnail(target_size, Image.Resampling.LANCZOS)
            return crop
