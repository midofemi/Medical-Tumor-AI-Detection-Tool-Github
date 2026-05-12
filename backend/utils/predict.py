"""
predict.py
──────────
Model inference utilities.
"""

import numpy as np
from backend.utils.logger import logger

# Must match the output layer's class ordering in the trained model
CLASSES = ["No Tumor", "Tumor", "Benign", "Malignant", "Normal"]


def predict_image(model, img: np.ndarray) -> tuple:
    """
    Run inference on a preprocessed image batch and return the top prediction.

    Args:
        model : Loaded tf.keras.Model.
        img   : 4-D float32 array of shape (1, H, W, C), values in [0, 1].

    Returns:
        tuple:
            label      (str)   – predicted class name.
            confidence (float) – probability of the predicted class.
            all_probs  (dict)  – {class_name: probability} for all classes.
    """
    try:
        # verbose=0 suppresses the TF progress bar for single-image inference
        preds = model.predict(img, verbose=0)
        idx   = int(np.argmax(preds))

        label      = CLASSES[idx]
        confidence = float(np.max(preds))
        all_probs  = {cls: float(preds[0][i]) for i, cls in enumerate(CLASSES)}

        logger.info(f"Prediction: {label} ({confidence:.2%})")
        return label, confidence, all_probs

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

