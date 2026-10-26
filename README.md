# Vehicle Component Defect Detection (Cast Metal Parts)

An end-to-end Computer Vision system for automated quality inspection of cast metal components (such as submersible pump impellers). The application classifies parts as **OK** or **Defective**, computes confidence scores, and localizes defect regions using **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps.

---

## 1. Problem Statement
In automotive and industrial manufacturing, metal casting is prone to defects such as:
- Subsurface blowholes and gas porosity
- Shrinkage cavities
- Pinholes and surface cracks
- Inclusions and mold wash

Manual optical inspection is labor-intensive, fatigue-prone, and inconsistent across shifts. Automated defect detection powered by Deep Learning delivers:
- **Consistent Quality Assurance:** Real-time part inspection before downstream assembly.
- **Cost Reduction:** Scrap reduction and avoidance of machining already-defective castings.
- **Explainability:** Visual justification for automated decisions using Grad-CAM heatmaps so quality engineers can inspect defect hotspots.

---

## 2. Dataset
- **Source:** Kaggle — [Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (by ravirajsinh45).
- **Target Component:** Cast submersible pump impeller.
- **Image Specifications:** ~300×300 grayscale JPEG images captured under stable top-down manufacturing lighting conditions.
- **Problem Type:** Binary image classification (`def_front` vs. `ok_front`).
- **Expected Directory Structure:**
  ```text
  data/
  └── casting_data/
      └── casting_data/
          ├── train/
          │   ├── def_front/   # Defective training samples
          │   └── ok_front/    # Non-defective (OK) training samples
          └── test/
              ├── def_front/   # Defective test samples
              └── ok_front/    # Non-defective (OK) test samples
  ```

---

## 3. Project Architecture & Approach
The project follows a phased machine learning engineering lifecycle:

1. **Phase 1: Project Setup & EDA (`notebooks/01_eda.ipynb`)**
   - Class balance verification
   - Image resolution and channel validation
   - Pixel-intensity distribution comparisons between OK and Defective parts
   - Visualizing typical casting defect artifacts (cracks, pinholes, blemishes)

2. **Phase 2: Data Pipeline & Augmentation (`src/data_loader.py`)**
   - Resizing to standard backbone input dimensions (224×224)
   - 85/15 stratified train/validation split with fixed seed reproducibility
   - Data augmentation pipeline: horizontal/vertical flips, small rotations, brightness adjustments, and zooms to simulate minor camera/positioning jitter
   - TensorFlow `tf.data` pipeline optimization (`cache()`, `prefetch(AUTOTUNE)`)

3. **Phase 3: Model Architecture & Training (`src/model.py`, `notebooks/02_training.ipynb`)**
   - Designed for GPU execution (e.g., Google Colab)
   - Baseline Custom CNN vs. Pretrained Transfer Learning Backbones (MobileNetV2 / ResNet50)
   - Training callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
   - Model artifact export: `models/model.keras`

4. **Phase 4: Evaluation & Explainability (`src/evaluate.py`, `src/gradcam.py`)**
   - Quantitative evaluation on holdout test set (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Confusion matrix analysis (critical focus on minimizing False Negatives: defective parts passed as OK)
   - Grad-CAM heatmap extraction from the final convolutional layer to highlight defect regions

5. **Phase 5: Interactive Web App (`app/app.py`)**
   - Streamlit interface for uploading part images
   - Inference execution against `models/model.keras`
   - Side-by-side display of original image, defect prediction badge, confidence score, and Grad-CAM overlay

---

## 4. Repository Structure
```text
vehicle-defect-detection/
├── .gitignore               # Ignores large datasets, virtualenvs, and model weights
├── README.md                # Project documentation and guide
├── requirements.txt         # Pinned project dependencies
├── data/                    # Dataset directory (gitignored)
│   └── .gitkeep
├── models/                  # Saved model weights (model.keras, gitignored)
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis notebook
│   └── 02_training.ipynb    # Google Colab GPU model training notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # tf.data loading & augmentation pipelines
│   ├── model.py             # CNN & Transfer Learning model builders
│   ├── evaluate.py          # Metric calculations & confusion matrix plots
│   ├── gradcam.py           # Grad-CAM heatmap generation utilities
│   └── utils.py             # General helper functions (image I/O, plotting)
├── app/
│   └── app.py               # Streamlit web application for inference & Grad-CAM
└── reports/
    └── figures/             # Exported EDA & evaluation plots
        └── .gitkeep
```

---

## 5. How to Run

### 5.1 Environment Setup
1. Clone this repository and open the directory:
   ```bash
   cd vehicle-defect-detection
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 5.2 Download Dataset
1. Download the dataset from Kaggle:
   - [Kaggle Dataset Link](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
   - Or via Kaggle CLI: `kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product`
2. Extract the archive into the `data/` folder so the paths match:
   - `data/casting_data/casting_data/train/`
   - `data/casting_data/casting_data/test/`

### 5.3 Run Exploratory Data Analysis (EDA)
Launch Jupyter and open `notebooks/01_eda.ipynb`:
```bash
jupyter lab notebooks/01_eda.ipynb
```
Running all cells will generate and save summary charts into `reports/figures/`.

### 5.4 Train the Model (Google Colab Recommended)
1. Upload `notebooks/02_training.ipynb` to [Google Colab](https://colab.research.google.com/) with a GPU runtime (T4 GPU recommended).
2. Download or upload the dataset in Colab as guided in the notebook.
3. Train the model and export the resulting `model.keras`.
4. Place the trained file locally at:
   ```text
   models/model.keras
   ```

### 5.5 Launch the Streamlit Web Application
Run the local web app to inspect castings interactively:
```bash
streamlit run app/app.py
```
Upload any test casting image to view prediction and Grad-CAM explanation.

---

## 6. Evaluation Results (Placeholders)

*Note: Models will be evaluated on the holdout test set (`data/casting_data/casting_data/test/`) once training in Colab is completed.*

| Model Architecture | Test Accuracy | Precision (Defective) | Recall (Defective) | F1-Score | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **MobileNetV2 (Transfer)** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **ResNet50 (Transfer)** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

> **Key Quality Metric:** In manufacturing, **Recall (Defective)** is the primary optimization objective. A False Negative (shipping a defective casting to assembly) carries a far higher penalty than a False Positive (re-inspecting a good part).

---

## 7. Limitations
- **Single-Perspective Imaging:** Images capture only top-down, planar views of the impeller. Side defects or internal cavities cannot be detected without X-ray or multi-angle cameras.
- **Lighting Sensitivity:** The model is trained on controlled lighting; changes in factory illuminance, shadowing, or glare may degrade performance without additional brightness/contrast augmentations.
- **Grayscale Domain:** Because inputs are single-channel grayscale castings, defects involving subtle material discoloration or chemical oxidation may require multi-spectral imaging.

---

## 8. Future Work
- [ ] Implement multi-label defect classification (crack vs. pinhole vs. porosity).
- [ ] Add semantic segmentation (e.g., U-Net / YOLOv8-seg) for pixel-precise defect area quantification.
- [ ] Export optimized inference graphs using ONNX Runtime / TensorFlow Lite for edge deployment on factory line cameras (e.g., Raspberry Pi or Jetson Nano).
- [ ] Integrate an active learning feedback loop allowing line operators to flag misclassified castings and retrain the model.
