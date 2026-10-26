"""Smoke test script for vehicle-defect-detection.

Builds both Baseline CNN and Transfer Learning (MobileNetV2) models,
verifies output prediction shapes, and validates Grad-CAM heatmap generation
and visual overlay on dummy image inputs without requiring dataset files.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Handle protobuf environment fallback for macOS/ARM if needed
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from src.model import build_baseline_cnn, build_transfer_model
from src.gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_gradcam


def run_smoke_tests():
    print("=" * 60)
    print("🔍 RUNNING LOCAL SMOKE TESTS — VEHICLE DEFECT DETECTION")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Executable:  {sys.executable}\n")

    # Generate dummy input: shape (1, 224, 224, 3) in [0, 255]
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    dummy_display = dummy_input[0].astype(np.uint8)

    # ---------------------------------------------------------
    # TEST 1: Baseline Custom CNN
    # ---------------------------------------------------------
    print("1️⃣ Testing Baseline Custom CNN...")
    baseline_model = build_baseline_cnn(input_shape=(224, 224, 3))
    base_preds = baseline_model(dummy_input, training=False)
    
    assert base_preds.shape == (1, 1), f"Unexpected baseline output shape: {base_preds.shape}"
    pred_val = float(base_preds[0][0])
    print(f"   ✓ Model output shape: {base_preds.shape} (Prediction: {pred_val:.4f})")

    base_conv = find_last_conv_layer(baseline_model)
    print(f"   ✓ Last Conv2D layer identified: '{base_conv}'")

    base_heatmap = make_gradcam_heatmap(dummy_input, baseline_model, last_conv_layer_name=base_conv)
    assert base_heatmap.ndim == 2, f"Heatmap must be 2D array, got {base_heatmap.ndim}D"
    print(f"   ✓ Grad-CAM heatmap computed successfully (Shape: {base_heatmap.shape}, Range: [{base_heatmap.min():.2f}, {base_heatmap.max():.2f}])")

    base_overlay, _ = overlay_gradcam(dummy_display, base_heatmap)
    assert base_overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {base_overlay.shape}"
    print(f"   ✓ Grad-CAM visual overlay rendered (Shape: {base_overlay.shape})\n")

    # ---------------------------------------------------------
    # TEST 2: Transfer Learning Model (MobileNetV2)
    # ---------------------------------------------------------
    print("2️⃣ Testing Transfer Learning Model (MobileNetV2 Backbone)...")
    transfer_model = build_transfer_model(input_shape=(224, 224, 3), backbone_name="MobileNetV2")
    trans_preds = transfer_model(dummy_input, training=False)
    
    assert trans_preds.shape == (1, 1), f"Unexpected transfer output shape: {trans_preds.shape}"
    trans_val = float(trans_preds[0][0])
    print(f"   ✓ Model output shape: {trans_preds.shape} (Prediction: {trans_val:.4f})")

    trans_conv = find_last_conv_layer(transfer_model)
    print(f"   ✓ Last Conv2D layer identified in backbone: '{trans_conv}'")

    trans_heatmap = make_gradcam_heatmap(dummy_input, transfer_model, last_conv_layer_name=trans_conv)
    assert trans_heatmap.ndim == 2, f"Heatmap must be 2D array, got {trans_heatmap.ndim}D"
    print(f"   ✓ Grad-CAM heatmap computed successfully (Shape: {trans_heatmap.shape}, Range: [{trans_heatmap.min():.2f}, {trans_heatmap.max():.2f}])")

    trans_overlay, _ = overlay_gradcam(dummy_display, trans_heatmap)
    assert trans_overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {trans_overlay.shape}"
    print(f"   ✓ Grad-CAM visual overlay rendered (Shape: {trans_overlay.shape})\n")

    print("=" * 60)
    print("✅ ALL SMOKE TESTS PASSED! Both models & Grad-CAM pipeline verified.")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_tests()
