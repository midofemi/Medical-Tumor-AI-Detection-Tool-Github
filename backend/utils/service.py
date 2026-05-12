"""
service.py
──────────
Service layer for tumor detection logic.
Handles model lazy-loading, image preprocessing, prediction, and GradCAM.
"""

import cv2
import numpy as np
import io
from backend.models.loader import load_model
from backend.utils.preprocess import preprocess_image
from backend.utils.predict import predict_image
#from backend.utils.gradcam import get_gradcam_heatmap, overlay_gradcam
from backend.utils.gradcam import (get_gradcam_plus_plus_heatmap, overlay_gradcam)
from backend.utils.logger import logger
import traceback


class TumorDetectionService:
    """
    Central service for all ML inference operations.
    """

    def __init__(self):
        self.model = None

    # ── Private Helpers ────────────────────────────────────────────────────────

    import traceback

    def _get_model(self):
        if self.model is None:
            logger.info("Loading ML model from HuggingFace Hub...")
            try:
                self.model = load_model()
            except Exception:
                logger.error("Full model loading traceback:")
                logger.error(traceback.format_exc())
                raise
            logger.info("Model loaded successfully.")
        return self.model

    def _find_last_conv_layer(self, model) -> str:
        """
        Walk the model layers in reverse and return the name of the last
        Conv2D-type layer.  Falls back to a hard-coded name only if none
        is found.

        Args:
            model: tf.keras.Model instance.

        Returns:
            str: Layer name to use for GradCAM.
        """
        #conv_types = ("Conv2D", "TrigConv2D", "DepthwiseConv2D", "SeparableConv2D")
        conv_types = ("Conv2D", "DepthwiseConv2D", "SeparableConv2D")
        for layer in reversed(model.layers):
            if layer.__class__.__name__ in conv_types:
                logger.info(f"GradCAM target layer resolved: '{layer.name}'")
                return layer.name
        # Last-resort fallback
        fallback = "conv2d"
        logger.warning(f"No convolutional layer found; using fallback '{fallback}'.")
        return fallback

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_and_predict(self, file_io):
        """
        Orchestrates the full inference pipeline:
          1. Preprocess the uploaded image.
          2. Run model prediction.
          3. Decode the original image for GradCAM overlay.

        Args:
            file_io: File-like object (BytesIO or UploadedFile) of the image.

        Returns:
            tuple: (label, confidence, all_probs, img_array, original_img)
                - label        (str)      : Predicted class name.
                - confidence   (float)    : Prediction probability [0, 1].
                - all_probs    (dict)     : Full class→probability mapping.
                - img_array    (ndarray)  : Batch-normalised image for GradCAM.
                - original_img (ndarray)  : Decoded BGR image for GradCAM overlay.
        """
        try:
            img_array = preprocess_image(file_io)

            model = self._get_model()
            label, confidence, all_probs = predict_image(model, img_array)

            # Re-read stream to decode the original image for visualization
            file_io.seek(0)
            file_bytes   = np.asarray(bytearray(file_io.read()), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            return label, confidence, all_probs, img_array, original_img

        except Exception as e:
            logger.error(f"Service execution failed: {e}")
            raise

    def generate_gradcam_image(self, img_array, original_img, layer_name: str = None):
        """
        Produces a GradCAM heatmap visualization overlaid on the original image.

        Dynamically resolves the target convolutional layer if `layer_name` is
        not provided or the named layer does not exist in the model.

        Args:
            img_array    (ndarray) : Preprocessed image batch.
            original_img (ndarray) : Original decoded BGR image.
            layer_name   (str|None): Convolutional layer name; auto-detected when None.

        Returns:
            ndarray: BGR image with GradCAM heatmap overlay.
        """
        try:
            model = self._get_model()

            # Auto-detect layer if not specified
            if layer_name is None:
                layer_name = "conv2d_1"
                #layer_name = self._find_last_conv_layer(model)
            else:
                # Verify the specified layer exists; fall back if not
                layer_names = [l.name for l in model.layers]
                if layer_name not in layer_names:
                    logger.warning(
                        f"Layer '{layer_name}' not found in model. "
                        "Auto-detecting convolutional layer instead."
                    )
                    #layer_name = self._find_last_conv_layer(model)
                    layer_name = "conv2d_1"

            #heatmap = get_gradcam_heatmap(model, img_array, layer_name)
            heatmap = get_gradcam_plus_plus_heatmap(model, img_array, layer_name)
            return overlay_gradcam(original_img, heatmap)

        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}")
            raise


# Singleton instance shared across the application
tumor_service = TumorDetectionService()
