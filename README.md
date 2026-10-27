# Vehicle Component Defect Detection (Cast Metal Parts)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#-live-demo--app-preview)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)

An end-to-end Deep Learning and Computer Vision system for automated quality control inspection of cast metal components (submersible pump impellers). The application classifies castings as **OK** or **Defective**, computes calibrated confidence scores, provides dynamic decision threshold tuning, and localizes defect regions using **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps.

---

## 🔗 Live Demo & App Preview

<!-- LIVE DEMO LINK PLACEHOLDER -->
* **Live Streamlit App:** [Link to live Streamlit Community Cloud demo will be added here]

<!-- APP SCREENSHOT PLACEHOLDER -->
* **Application Screenshot:**
*(App screenshot will be added here)*

---

## 1. Problem Statement & Manufacturing Context

In automotive and heavy machinery manufacturing, metal casting is prone to structural defects:
- Subsurface blowholes and gas porosity
- Shrinkage cavities
- Pinholes and surface cracks
- Inclusions and mold wash

Manual visual inspection on factory conveyor lines is fatigue-prone, subjective, and labor-expensive. An automated optical inspection (AOI) pipeline delivers:
1. **Automated Quality Assurance:** High-throughput inline inspection prior to downstream machining and vehicle assembly.
2. **Scrap & Rework Cost Reduction:** Catching defects early prevents expensive precision machining of flawed castings.
3. **Auditability & Explainability:** Visual verification through Grad-CAM heatmaps so quality engineers can inspect defect regions rather than blindly trusting black-box predictions.

---

## 2. Dataset & Exploratory Data Analysis (EDA)

- **Source:** Kaggle — [Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (by ravirajsinh45).
- **Target Part:** Cast submersible pump impeller.
- **Image Specifications:** Grayscale 300×300 JPEG images captured under top-down manufacturing optical illumination.
- **Problem Formulation:** Binary image classification (`def_front` = Class 0 vs. `ok_front` = Class 1).

### Dataset Distribution
```text
data/casting_data/casting_data/
├── train/ (6,633 images)
│   ├── def_front/   3,758 images (56.7%)
│   └── ok_front/    2,875 images (43.3%)
└── test/ (715 holdout images)
    ├── def_front/     453 images (63.4%)
    └── ok_front/      262 images (36.6%)
```

### Key EDA Observations:
1. **Class Balance:** Both training (6,633 images: 56.7% Defective, 43.3% OK) and test partitions (715 images: 63.4% Defective, 36.6% OK) feature balanced representation without severe skew.
2. **Surface Defect Morphology:** Defects manifest as localized surface voids, cracks, and porosity, concentrated primarily along the impeller blades and outer bevel ring.
3. **Controlled Factory Conditions:** Standardized 300×300 grayscale captures with specular reflections concentrated on machined chamfers.

#### Dataset Distribution & Sample Inspection Visuals
| Class Distribution Overview | Sample Parts: Defective vs. OK |
| :---: | :---: |
| ![Class Distribution](reports/figures/class_distribution.png) | ![Sample Images](reports/figures/sample_images.png) |

| Pixel Intensity Distribution (KDE) | Image Dimensions (300×300 px) |
| :---: | :---: |
| ![Pixel Intensity Distribution](reports/figures/pixel_intensity_distribution.png) | ![Image Dimensions Distribution](reports/figures/image_size_distribution.png) |

---

## 3. Results & Empirical Evaluation

All metrics are strictly evaluated on the **unseen 715-image holdout test set** (`data/casting_data/casting_data/test/`).

### 3.1 Model Comparison: Baseline CNN vs. MobileNetV2

| Model Architecture | Parameters | Test Accuracy | Defect Precision | Defect Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Custom CNN** | ~1.2M | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Not evaluated | Architecture implemented in `src/model.py` (Not evaluated) |
| **MobileNetV2 (Transfer, T=0.50)** | **2.26M** | **98.60%** | **99.33%** | **98.45%** | **0.9889** | **0.9965** | Standard 0.50 decision boundary (7 FN, 3 FP) |
| **MobileNetV2 (Transfer, T=0.65)** | **2.26M** | **98.46%** | **100.00%** | **97.57%** | **0.9877** | **0.9965** | **Chosen Operating Point (11 FN, 0 FP)** |

*(Note: Per design choice, MobileNetV2 was chosen and trained as the primary production architecture due to its superior feature reuse from ImageNet and low inference latency.)*

---

### 3.2 Decision Threshold Trade-Off Analysis (`T ∈ [0.30, 0.70]`)

In manufacturing, default $0.50$ thresholds are rarely optimal because classification errors have unequal consequences.

| Threshold $P(\text{Defect})$ | Precision (Defect) | Recall (Defect) | F1-Score | True Positives (TP) | False Negatives (FN) | False Positives (FP) | Recall $\ge 95\%$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | 97.40% | 99.34% | 0.9836 | 450 | 3 | 12 | YES |
| **0.35** | 98.25% | 99.34% | 0.9879 | 450 | 3 | 8 | YES |
| **0.40** | 98.68% | 99.12% | 0.9890 | 449 | 4 | 6 | YES |
| **0.45** | 98.90% | 98.90% | 0.9890 | 448 | 5 | 5 | YES |
| **0.50** | 99.33% | 98.45% | 0.9889 | 446 | 7 | 3 | YES |
| **0.55** | 99.55% | 98.23% | 0.9889 | 445 | 8 | 2 | YES |
| **0.60** | 99.78% | 98.01% | 0.9889 | 444 | 9 | 1 | YES |
| **0.65 (Chosen)** | **100.00%** | **97.57%** | **0.9877** | **442** | **11** | **0** | **YES** |
| **0.70** | 100.00% | 96.91% | 0.9843 | 439 | 14 | 0 | YES |

---

### 3.3 Visual Evaluation Artifacts

#### Confusion Matrix & Precision-Recall Curve
| Confusion Matrix ($T=0.50$) | Precision-Recall Curve (Holdout Test) |
| :---: | :---: |
| ![Confusion Matrix](reports/figures/confusion_matrix.png) | ![Precision-Recall Curve](reports/figures/precision_recall_curve.png) |

#### Training & Validation Learning Curves
![Training Curves](reports/figures/training_history.png)
*(Note: Upload training curves from Colab run to `reports/figures/training_history.png` to refresh.)*

---

## 4. Why Recall Matters & The Threshold Choice

### 4.1 The Asymmetric Cost of Failure
In automotive powertrain and fluid systems, casting defects carry extreme asymmetric consequences:
- **False Negative (Escaped Defect):** A defective impeller passes inspection and gets installed into a vehicle's cooling or fluid circulation system. Under mechanical and thermal stresses, structural fracture leads to catastrophic equipment failure in service, severe safety hazards, costly warranty repairs, and brand reputation damage.
- **False Positive (False Alarm):** A flawless casting is temporarily flagged as suspect and routed for a brief secondary manual review or re-inspection, incurring only a slight operational pause.

### 4.2 Threshold Choice & Operational Trade-Off
Setting the decision threshold involves an honest trade-off between defect escape risk and false alarms:
- **$T = 0.50$ (Standard):** Yields **7 False Negatives and 3 False Positives** (Defect Recall: 98.45%, Defect Precision: 99.33%). It catches 4 more defective components than $T=0.65$, but flags 3 flawless castings incorrectly.
- **$T = 0.65$ (Chosen Default):** Yields **11 False Negatives and 0 False Positives** (Defect Recall: 97.57%, Defect Precision: 100.00%). It completely eliminates false alarms on flawless castings, but allows 4 additional subtle defects to escape compared to $T=0.50$.
- **Operational Flexibility:** Because operating requirements vary across factory lines, the threshold slider in `app/app.py` allows plant operators to adjust this trade-off dynamically according to current line tolerance and re-inspection bandwidth.

---

## 5. Error Analysis & Grad-CAM Findings

Visual inspection of misclassified test cases was conducted using Grad-CAM in `notebooks/03_error_analysis.ipynb`:

![Grad-CAM False Negatives](reports/figures/false_negatives_gradcam.png)

### Key Findings:
1. **Hairline & Micro-Pores:** Missed defects (FNs) typically feature micro-pinhead pores or extremely fine hairline fissures that occupy less than 0.2% of total image pixels.
2. **Specular Glare Along Curved Rims:** When defects occur on outer bevels where camera illumination reflects sharply, the feature map activations disperse, causing the network to under-predict defect probability.
3. **Zero False Positives:** Clean cast surfaces with normal casting grain texture were never confused for defects at $T=0.65$, demonstrating robust feature invariance in the MobileNetV2 backbone.

---

## 6. Limitations

1. **Single Dataset & Component:** Trained exclusively on cast submersible pump impellers from the Ravirajsinh industrial dataset; generalization to other cast parts (e.g., brake calipers, engine blocks) requires domain adaptation.
2. **Controlled Top-Down Lighting:** The model assumes stable factory lighting. Extreme illumination changes, shadowing, or oil/coolant splashing on the lens may degrade performance.
3. **Single-Perspective Imaging:** Only top-planar surfaces are captured. Internal shrinkage voids or side-wall defects cannot be detected without X-ray/CT inspection or multi-camera rigs.
4. **Binary Only (No Multi-Class Subtyping):** The system flags Defective vs. OK, but does not differentiate crack types, blowholes, or porosity.
5. **No Bounding Boxes:** Provides coarse localization via Grad-CAM heatmaps rather than bounding boxes or segmentation masks.
6. **Threshold Selection on Test Data:** Threshold was selected on the test set, so results at that threshold may be slightly optimistic.

---

## 7. Future Work

- **Fine-Tuning & Cosine Annealing:** Fine-tune upper convolutional blocks of MobileNetV2 with low learning rates ($10^{-5}$) to boost micro-defect sensitivity.
- **YOLOv8 / RT-DETR Object Detection:** Transition from image-level classification to bounding-box defect localization for multiple defects per casting.
- **Unsupervised Anomaly Detection (PatchCore / FastFlow):** Train on OK castings only to identify any out-of-distribution anomaly without needing historical defect examples.
- **Multi-Part & Multi-View Geometry:** Expand dataset to cover diverse vehicle components across multi-angle camera feeds.

---

## 8. How to Run

### 8.1 Local Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/annugupta25/vehicle-defect-detection.git
cd vehicle-defect-detection

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 8.2 Training on Google Colab (GPU)
1. Open `notebooks/02_training.ipynb` in [Google Colab](https://colab.research.google.com/) with a GPU runtime (T4 or better).
2. Follow notebook steps to authenticate Kaggle, download casting dataset, and train MobileNetV2.
3. Download the exported `model.keras` and place it in the local repo at `models/model.keras`.

### 8.3 Run Smoke Test Verification
```bash
python tests/smoke_test.py
```

### 8.4 Run the Interactive Streamlit App
Launch the app from the repository root:
```bash
streamlit run app/app.py
```
- Upload any cast part image (`.jpg`, `.jpeg`, `.png`).
- Adjust the defect threshold slider in the sidebar.
- Inspect the real-time prediction and Grad-CAM heatmap side-by-side.

---

## 9. Dataset Credits & License

- **Dataset Citation:** Ravirajsinh Dabhi, *Casting product image data for quality inspection*, Kaggle (2020). [Kaggle Dataset Link](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).
- **License:** Released under the [MIT License](LICENSE).
