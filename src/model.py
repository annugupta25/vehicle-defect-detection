"""Model architecture definitions for casting defect detection.

Provides builders for:
1. Baseline Custom CNN (lightweight, interpretable conv architecture)
2. Transfer Learning Model (MobileNetV2 / ResNet50 backbone pretrained on ImageNet)
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Builds a custom convolutional neural network for binary defect detection.

    Architecture:
      - 4 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPooling2D)
      - GlobalAveragePooling2D to minimize parameter count and overfitting
      - Dense classification head with Dropout and Sigmoid activation

    Args:
        input_shape: Shape of the input image (height, width, channels).
        dropout_rate: Dropout probability before the final dense layer.

    Returns:
        keras.Model: Uncompiled Keras Model instance.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Rescale pixel values from [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4 (Target layer for Grad-CAM)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_target")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_casting_cnn")
    return model


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    backbone_name: str = "MobileNetV2",
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Builds a transfer learning model using a pretrained ImageNet backbone.

    Args:
        input_shape: Shape of input image (height, width, channels).
        backbone_name: "MobileNetV2" or "ResNet50".
        dropout_rate: Dropout rate for classification head.

    Returns:
        keras.Model: Uncompiled Keras Model with frozen backbone.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    if backbone_name == "MobileNetV2":
        # Preprocessing expects [-1, 1] range
        preprocess_input = keras.applications.mobilenet_v2.preprocess_input
        x = preprocess_input(inputs)
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name == "ResNet50":
        preprocess_input = keras.applications.resnet50.preprocess_input
        x = preprocess_input(inputs)
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze base model weights for feature extraction
    base_model.trainable = False
    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{backbone_name.lower()}_casting_model")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """Compiles the model with Adam optimizer, binary crossentropy, and monitoring metrics.

    Args:
        model: Keras Model to compile.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        keras.Model: Compiled model.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model
