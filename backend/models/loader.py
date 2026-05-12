"""
loader.py
─────────
Defines TrigConv2D (a custom Keras layer that uses trigonometric kernels)
and the load_model() factory which downloads the pre-trained model from
HuggingFace Hub on first use.
"""

import tensorflow as tf
import numpy as np

class TrigConv2D(tf.keras.layers.Layer):
    """
    Custom Conv2D variant whose kernels are fixed sinusoidal / cosine patterns.

    Even-indexed filters use sin, odd-indexed filters use cos, based on a
    2-D grid spanning [-1, 1] in both spatial dimensions.

    Args:
        filters      : Number of output filters.
        kernel_size  : Square kernel side length.
        frequency    : Frequency of the trig pattern (default 1.0).
    """

    def __init__(self, filters: int, kernel_size: int, frequency: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.frequency   = frequency

    def build(self, input_shape):
        # Call super().build() FIRST so Keras marks the layer as built and
        # registers it in the graph before we define our custom weight.
        super().build(input_shape)

        kernels = []
        x = np.linspace(-1, 1, self.kernel_size)
        y = np.linspace(-1, 1, self.kernel_size)
        x_grid, y_grid = np.meshgrid(x, y)

        for i in range(self.filters):
            # Alternate between sin and cos patterns per filter
            if i % 2 == 0:
                kernel = np.sin(self.frequency * (x_grid + y_grid))
            else:
                kernel = np.cos(self.frequency * (x_grid + y_grid))

            # Shape: (kernel_size, kernel_size, in_channels, 1)
            kernel = kernel[:, :, np.newaxis, np.newaxis]
            kernel = np.repeat(kernel, input_shape[-1], axis=2)
            kernels.append(kernel)

        self.kernel = tf.constant(
            np.concatenate(kernels, axis=3), dtype=tf.float32
        )

    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding="SAME")

    def get_config(self):
        """Serialisation support — required for model save/load."""
        config = super().get_config()
        config.update({
            "filters":     self.filters,
            "kernel_size": self.kernel_size,
            "frequency":   self.frequency,
        })
        return config

def load_model() -> tf.keras.Model:

    """
    Here we are just loading our model from hugging face. In this case, we will be using baseline_cnn.keras which was already trained and uploaded 
    to HuggingFace Hub. 
    
    """
    from backend.config import HF_REPO_ID, MODEL_FILENAME
    from huggingface_hub import hf_hub_download
    import os
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False,
    )
