"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Produces visual explanations for CNN decisions by computing gradients of the
predicted class score with respect to feature maps of the final convolutional layer.
Supports both custom CNNs and nested transfer learning backbones (e.g. MobileNetV2).
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import tensorflow as tf
try:
    import keras
except ImportError:
    from tensorflow import keras


def find_last_conv_layer(model: keras.Model) -> str:
    """Finds the name of the last 2D Convolutional layer in a Keras model.

    Searches backwards through top-level layers and any nested Functional submodels.

    Args:
        model: Keras Model instance.

    Returns:
        str: Name of the last Conv2D layer found.
    """
    # 1. Search top-level layers in reverse
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # 2. Search inside nested submodels (e.g., MobileNetV2 functional block)
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, keras.layers.Conv2D):
                    return sublayer.name

    raise ValueError("Could not find any Conv2D layer in the model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Generates a Grad-CAM heatmap for a single image array.

    Compatible with both standard sequential/functional CNNs and models with
    nested pretrained functional backbones (e.g., MobileNetV2).

    Args:
        img_array: Input tensor of shape (1, height, width, 3).
        model: Trained Keras model.
        last_conv_layer_name: Name of target Conv layer. If None, auto-detected.
        pred_index: Target class index to explain (0: Defective, 1: OK).
                    If None, explains the predicted class.

    Returns:
        np.ndarray: 2D numpy array [0, 1] representing activation heatmap.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Check if target layer is a top-level layer or inside a nested submodel
    is_nested = False
    nested_layer = None
    nested_idx = -1

    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except (ValueError, AttributeError):
        # Look inside submodules
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "layers"):
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    nested_layer = layer
                    nested_idx = i
                    is_nested = True
                    break
                except (ValueError, AttributeError):
                    continue

    if not is_nested:
        # Standard functional/sequential model (e.g. Baseline CNN)
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            prob = predictions[:, 0]
            # If pred_index is None, explain the winning class
            target_class = pred_index if pred_index is not None else tf.cast(prob >= 0.5, tf.int32).numpy()[0]
            # Binary loss: if target is Defective (0), maximize (1 - prob); if OK (1), maximize prob
            loss = prob if target_class == 1 else (1.0 - prob)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

    else:
        # Nested model architecture (e.g. MobileNetV2 backbone called inside model)
        # 1. Submodel for feature extraction inside the nested backbone
        feature_extractor = keras.Model(
            inputs=nested_layer.inputs,
            outputs=[target_layer.output, nested_layer.output],
        )

        # 2. Classifier head submodel representing layers after the nested backbone
        classifier_input = keras.Input(shape=nested_layer.output.shape[1:])
        x_clf = classifier_input
        for layer in model.layers[nested_idx + 1:]:
            x_clf = layer(x_clf)
        classifier_model = keras.Model(inputs=classifier_input, outputs=x_clf)

        # 3. Compute gradients end-to-end
        with tf.GradientTape() as tape:
            # Pass input through any preprocessing layers prior to nested backbone
            x_in = img_array
            for layer in model.layers[:nested_idx]:
                if isinstance(layer, keras.layers.InputLayer):
                    continue
                x_in = layer(x_in)

            conv_outputs, base_out = feature_extractor(x_in)
            tape.watch(conv_outputs)

            predictions = classifier_model(base_out)
            prob = predictions[:, 0]
            target_class = pred_index if pred_index is not None else tf.cast(prob >= 0.5, tf.int32).numpy()[0]
            loss = prob if target_class == 1 else (1.0 - prob)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only retain activations that contribute positively to the target class
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

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
