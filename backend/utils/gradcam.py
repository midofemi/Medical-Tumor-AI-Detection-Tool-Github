"""
gradcam.py
──────────
Gradient-weighted Class Activation Mapping (GradCAM) utilities.

Provides:
  - get_gradcam_heatmap  : compute the normalised heatmap array.
  - overlay_gradcam      : blend the heatmap onto the original image.
"""

import numpy as np
import tensorflow as tf
import cv2
from backend.utils.logger import logger


def get_gradcam_plus_plus_heatmap(model, img_array, layer_name: str = "conv2d") -> np.ndarray:
    try:
        inputs = tf.keras.Input(shape=img_array.shape[1:])

        x = inputs
        conv_output = None

        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_output = x

        if conv_output is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")

        grad_model = tf.keras.Model(
            inputs=inputs,
            outputs=[conv_output, x]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array, training=False)

            if isinstance(preds, list):
                preds = preds[0]

            class_idx = tf.argmax(preds[0])
            class_score = preds[:, class_idx]

        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            raise RuntimeError(
                f"GradCAM++: gradient w.r.t. layer '{layer_name}' is None."
            )

        conv_outputs = conv_outputs[0]
        grads = grads[0]

        first_derivative = grads
        second_derivative = tf.square(grads)
        third_derivative = tf.pow(grads, 3)

        global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))

        alpha_num = second_derivative
        alpha_denom = (
            2.0 * second_derivative
            + third_derivative * global_sum[tf.newaxis, tf.newaxis, :]
        )

        alpha_denom = tf.where(
            alpha_denom != 0.0,
            alpha_denom,
            tf.ones_like(alpha_denom)
        )

        alphas = alpha_num / alpha_denom

        weights = tf.reduce_sum(
            alphas * tf.maximum(first_derivative, 0.0),
            axis=(0, 1)
        )

        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0).numpy()

        # Suppress weak background activations
        heatmap = np.maximum(heatmap - np.percentile(heatmap, 60), 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    except Exception as e:
        logger.error(f"GradCAM++ heatmap error: {e}")
        raise


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Blend a GradCAM heatmap onto the original image.

    Uses cv2.addWeighted so both the heatmap and original contribute
    proportionally (heatmap * alpha + original * (1-alpha)), preventing
    the output from becoming over-bright or washed out.

    Args:
        original_img : Decoded BGR image (H, W, 3).
        heatmap      : 2-D float array from get_gradcam_heatmap.
        alpha        : Heatmap intensity weight (default 0.4).

    Returns:
        np.ndarray: Blended BGR image as uint8.
    """
    try:
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)

        # Apply JET colour-map (OpenCV produces BGR output)
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend in BGR space (both images are BGR at this point)
        blended_bgr = cv2.addWeighted(heatmap_colored, alpha, original_img, 1.0 - alpha, 0)

        # Convert to RGB before returning so that callers (PNG encoder, browser)
        # receive correct colours — browsers interpret PNG/JPEG as RGB, not BGR.
        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        logger.error(f"GradCAM overlay error: {e}")
        raise

