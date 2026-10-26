"""Utility helper functions for image I/O, preprocessing, and path management."""

import os
from typing import Tuple, Union
import numpy as np
from PIL import Image
import tensorflow as tf


def preprocess_image_for_model(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads and preprocesses an image for model inference.

    Args:
        image: Path to image file, PIL Image, or numpy array.
        target_size: Desired model input dimensions (height, width).

    Returns:
        tuple: (batch_array, display_array)
          - batch_array: shape (1, H, W, 3) float32 for model input
          - display_array: shape (H, W, 3) uint8 for UI / Grad-CAM rendering
    """
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Resize to target input dimensions
    resized_img = pil_img.resize(target_size)
    display_array = np.array(resized_img, dtype=np.uint8)

    # Convert to float array with batch dimension (1, H, W, 3)
    batch_array = np.expand_dims(display_array.astype(np.float32), axis=0)

    return batch_array, display_array


def check_dataset_paths(
    base_dir: str = "data/casting_data/casting_data",
) -> dict:
    """Checks whether the expected Kaggle dataset directories exist.

    Args:
        base_dir: Path to unzipped casting dataset root.

    Returns:
        dict: Status information including paths and existence flags.
    """
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_def = os.path.join(train_dir, "def_front")
    train_ok = os.path.join(train_dir, "ok_front")
    test_def = os.path.join(test_dir, "def_front")
    test_ok = os.path.join(test_dir, "ok_front")

    return {
        "base_exists": os.path.exists(base_dir),
        "train_exists": os.path.exists(train_dir),
        "test_exists": os.path.exists(test_dir),
        "train_def_count": len(os.listdir(train_def)) if os.path.exists(train_def) else 0,
        "train_ok_count": len(os.listdir(train_ok)) if os.path.exists(train_ok) else 0,
        "test_def_count": len(os.listdir(test_def)) if os.path.exists(test_def) else 0,
        "test_ok_count": len(os.listdir(test_ok)) if os.path.exists(test_ok) else 0,
    }
