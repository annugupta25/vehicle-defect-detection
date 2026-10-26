"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Produces visual explanations for CNN decisions by computing gradients of the
predicted class score with respect to feature maps of the final convolutional layer.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model: keras.Model) -> str:
    """Finds the name of the last 2D Convolutional layer in a Keras model.

    Args:
        model: Keras Model instance.

    Returns:
        str: Name of the last Conv2D layer.
    """
    for layer in reversed(model.layers):
        # Check direct Conv2D layer
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # If model has nested layers (e.g., Transfer Learning Functional submodels)
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, keras.layers.Conv2D):
                    return sublayer.name

    raise ValueError("Could not find any Conv2D layer in the model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generates a Grad-CAM heatmap for a single image array.

    Args:
        img_array: Batch tensor of shape (1, height, width, channels), scaled as expected by model.
        model: Trained Keras model.
        last_conv_layer_name: Optional name of target Conv layer. If None, automatically detected.

    Returns:
        np.ndarray: 2D numpy array with values in [0, 1] representing activation intensity.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Build a submodel that outputs both the target conv feature maps and the final predictions
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output],
        )
    except (ValueError, AttributeError):
        # Fallback if layer is nested inside a sub-module
        nested_model = None
        for layer in model.layers:
            if hasattr(layer, "layers"):
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    nested_model = layer
                    break
                except ValueError:
                    continue

        if nested_model is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

        grad_model = keras.models.Model(
            inputs=nested_model.inputs,
            outputs=[target_layer.output, nested_model.output],
        )

    # Record operations for gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification with sigmoid output:
        loss = predictions[:, 0]

    # Compute gradients of class score with respect to feature map activations
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling across spatial dimensions to get channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map channels by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features that contribute positively to the class score
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Superimposes Grad-CAM heatmap onto the original image.

    Args:
        original_img: Original image array (uint8, 0-255 range).
        heatmap: 2D float array [0, 1] returned by make_gradcam_heatmap.
        alpha: Blending weight for the heatmap overlay (0.0 = original, 1.0 = pure heatmap).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        tuple: (superimposed_img, colored_heatmap) as uint8 RGB numpy arrays.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert normalized heatmap [0, 1] to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap (yields BGR)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Ensure original image is RGB
    if len(original_img.shape) == 2 or original_img.shape[-1] == 1:
        base_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = original_img.copy()

    # Blend original image and heatmap
    superimposed = cv2.addWeighted(base_rgb, 1 - alpha, colored_heatmap_rgb, alpha, 0)
    return superimposed, colored_heatmap_rgb
