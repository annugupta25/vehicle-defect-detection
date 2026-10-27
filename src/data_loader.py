"""Data loading and augmentation pipeline for cast metal defect detection.

This module provides functions to load train, validation, and test datasets
using TensorFlow's `image_dataset_from_directory`, along with Keras data
augmentation layers (flip, rotation, brightness, zoom).
"""

import os
from typing import Tuple, Optional
import tensorflow as tf
try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers


def get_augmentation_pipeline() -> keras.Sequential:
    """Creates a Keras Sequential model for real-time data augmentation.

    Augmentations include:
      - Random horizontal and vertical flips
      - Small random rotations (simulates part positioning variations)
      - Random brightness adjustments (simulates minor lighting variations)
      - Random zoom in/out

    Returns:
        keras.Sequential: A pipeline of Keras image preprocessing layers.
    """
    augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal_and_vertical",
                name="random_flip"
            ),
            layers.RandomRotation(
                factor=0.10,
                fill_mode="reflect",
                name="random_rotation"
            ),
            layers.RandomBrightness(
                factor=0.15,
                value_range=(0, 255),
                name="random_brightness"
            ),
            layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                fill_mode="reflect",
                name="random_zoom"
            ),
        ],
        name="data_augmentation",
    )
    return augmentation


def load_datasets(
    data_dir: str = "data/casting_data/casting_data",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
    color_mode: str = "rgb",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], list]:
    """Loads train, validation, and test datasets from directory structure.

    Directory structure expected:
      data_dir/
        ├── train/
        │   ├── def_front/
        │   └── ok_front/
        └── test/
            ├── def_front/
            └── ok_front/

    Args:
        data_dir: Base directory containing 'train' and 'test' subfolders.
        img_size: Target image dimensions (height, width). Default (224, 224).
        batch_size: Number of images per batch. Default 32.
        validation_split: Fraction of train data reserved for validation (0.15 = 85/15 split).
        seed: Random seed for deterministic train/validation splitting.
        color_mode: "rgb" (3 channels, recommended for transfer learning) or "grayscale".

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Train directory not found at: {train_dir}\n"
            f"Please ensure the Kaggle dataset is unzipped into '{data_dir}'."
        )

    # Load 85% for Training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode="binary",
        shuffle=True,
    )

    # Load 15% for Validation (using identical seed and split)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names

    # Load holdout Test set if folder exists
    test_ds = None
    if os.path.exists(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode="binary",
            shuffle=False,
        )

    return train_ds, val_ds, test_ds, class_names


def configure_performance(
    ds: tf.data.Dataset,
    is_training: bool = False,
    augment: bool = False,
    augmentation_pipeline: Optional[keras.Sequential] = None,
) -> tf.data.Dataset:
    """Applies caching, optional augmentation, and prefetching for optimal GPU throughput.

    Args:
        ds: The input tf.data.Dataset.
        is_training: Whether the dataset is used for training (enables shuffling).
        augment: Whether to apply the augmentation pipeline.
        augmentation_pipeline: Keras Sequential augmentation model.

    Returns:
        tf.data.Dataset: Optimized dataset ready for model consumption.
    """
    # Cache images in memory after first epoch to eliminate disk I/O bottleneck
    ds = ds.cache()

    if is_training:
        ds = ds.shuffle(buffer_size=1000)

    if augment:
        if augmentation_pipeline is None:
            augmentation_pipeline = get_augmentation_pipeline()
        ds = ds.map(
            lambda x, y: (augmentation_pipeline(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch batches in background while GPU is training on current batch
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
