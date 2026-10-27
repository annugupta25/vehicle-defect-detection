"""Evaluation utilities for casting defect classification models.

Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
plots confusion matrices, and plots training history curves.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
)
import tensorflow as tf
try:
    import keras
except ImportError:
    from tensorflow import keras


def evaluate_dataset(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates the model on a given dataset and returns comprehensive metrics.

    Args:
        model: Trained Keras model.
        test_ds: Holdout test tf.data.Dataset yielding (images, labels).
        threshold: Decision boundary probability (default: 0.5).

    Returns:
        dict: Dictionary containing predictions, true labels, and scalar metrics.
    """
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds.flatten())
        y_true_list.append(labels.numpy().flatten())

    y_true = np.concatenate(y_true_list)
    y_pred_probs = np.concatenate(y_pred_probs_list)
    y_pred = (y_pred_probs >= threshold).astype(int)

    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_probs)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    results = {
        "y_true": y_true,
        "y_pred_probs": y_pred_probs,
        "y_pred": y_pred,
        "f1_score": f1,
        "roc_auc": auc,
        "classification_report": report_dict,
    }
    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ["Defective", "OK"],
    save_path: Optional[str] = "reports/figures/confusion_matrix.png",
) -> plt.Figure:
    """Plots and optionally saves a styled confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Names for class 0 and class 1.
        save_path: Filepath to save the figure, or None.

    Returns:
        plt.Figure: The created Matplotlib figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Casting Inspection", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = "reports/figures/training_history.png",
) -> plt.Figure:
    """Plots training and validation curves for Loss and Accuracy.

    Args:
        history: Keras History object returned by model.fit().
        save_path: Filepath to save the figure, or None.

    Returns:
        plt.Figure: The created Matplotlib figure.
    """
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss Curve
    axes[0].plot(epochs, history.history["loss"], label="Train Loss", color="#d9534f", lw=2)
    if "val_loss" in history.history:
        axes[0].plot(epochs, history.history["val_loss"], label="Val Loss", color="#0275d8", lw=2, linestyle="--")
    axes[0].set_title("Loss over Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy Curve
    axes[1].plot(epochs, history.history["accuracy"], label="Train Accuracy", color="#5cb85c", lw=2)
    if "val_accuracy" in history.history:
        axes[1].plot(epochs, history.history["val_accuracy"], label="Val Accuracy", color="#f0ad4e", lw=2, linestyle="--")
    axes[1].set_title("Accuracy over Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history saved to: {save_path}")

    return fig
