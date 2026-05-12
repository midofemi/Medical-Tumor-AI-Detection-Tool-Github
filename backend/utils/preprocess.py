"""
preprocess.py
─────────────
"""

import cv2
import numpy as np
from backend.config import IMG_SIZE
from backend.utils.logger import logger


def preprocess_image(uploaded_file) -> np.ndarray:
    """
    Decode a raw file stream into a model-ready batch tensor.

    Processing steps:
      1. Decode bytes → BGR image (OpenCV).
      2. Resize to (IMG_SIZE × IMG_SIZE).
      3. Cast to float32 and normalise pixel values to [0, 1].
      4. Expand to a batch of 1  →  shape (1, H, W, C).

    Args:
        uploaded_file: File-like object (BytesIO from FastAPI).

    Returns:
        np.ndarray: float32 array of shape (1, IMG_SIZE, IMG_SIZE, 3).

    Raises:
        ValueError: If the file is empty or the image cannot be decoded.
    """
    try:
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        if file_bytes.size == 0:
            raise ValueError("Empty file uploaded.")

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image — unsupported format or corrupt file.")

        # Resize to the model's expected spatial dimensions
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # float32 (not float64) halves memory and matches TF/Keras expectations
        img = img.astype(np.float32) / 255.0

        return np.expand_dims(img, axis=0)  # (1, H, W, C)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

