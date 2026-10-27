# Streamlit Community Cloud Deployment Guide

This guide provides the exact steps to deploy the **Vehicle Component Defect Detection** web application on [Streamlit Community Cloud](https://streamlit.io/cloud) for free public access.

---

## 1. Prerequisites Checklist

- [x] **Repository on GitHub:** Public repository at `https://github.com/annugupta25/vehicle-defect-detection.git`.
- [x] **Model Weights Included:** `models/model.keras` is **11.06 MB** (well under GitHub's 100 MB limit) and is directly tracked in Git. No Hugging Face or external cloud storage downloads required.
- [x] **Deployment-Ready Dependencies:** `requirements.txt` specifies `opencv-python-headless` (to avoid missing X11/GL system libraries in Linux containers) and pinned `tensorflow==2.16.2` with `numpy<2.0.0`.
- [x] **Path Independence:** `app/app.py` resolves the model path relative to the project root using `Path(__file__).resolve().parent.parent`.

---

## 2. Step-by-Step Deployment

### Step 1: Sign In to Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/).
2. Log in with your **GitHub account** (`annugupta25`).

### Step 2: Deploy New App
1. Click the **"New app"** (or **"Create app"**) button in the top right.
2. Select **"Use existing repo"**.

### Step 3: Configure App Settings
Fill in the deployment form with the following parameters:

| Field | Configuration Value | Notes |
| :--- | :--- | :--- |
| **Repository** | `annugupta25/vehicle-defect-detection` | Select your repository |
| **Branch** | `main` | Default production branch |
| **Main file path** | `app/app.py` | Path to Streamlit app script |
| **App URL (optional)** | `vehicle-defect-detection` | Choose custom subdomain if available |

### Step 4: Advanced Settings (Python Version)
1. Click **"Advanced settings"** before clicking Deploy.
2. In the **Python version** dropdown, select **`3.11`** (recommended for TensorFlow 2.16+).
3. Secrets are not required (no API keys needed for inference).
4. Click **"Save"**.

### Step 5: Click "Deploy!"
1. Click the **"Deploy!"** button.
2. Streamlit Cloud will spin up a Debian container, clone the repo, install `requirements.txt`, and start the app.
3. The build log will show dependency installation (`tensorflow`, `keras`, `opencv-python-headless`, etc.).

---

## 3. Post-Deployment Verification

1. Once the app is running, verify that:
   - The sidebar displays: `models/model.keras found`.
   - The decision threshold slider defaults to `0.65`.
   - Uploading a test image from `data/casting_data/casting_data/test/def_front/` produces an immediate **DEFECTIVE (Reject)** badge and Grad-CAM heatmap.
   - Uploading a test image from `data/casting_data/casting_data/test/ok_front/` produces an **OK (Pass)** badge.
2. Copy your public Streamlit app URL and update the placeholder link in `README.md`.

---

## 4. Troubleshooting & Best Practices

- **Container Memory Limit:** Streamlit Community Cloud provides 1 GB of RAM on free tier. Since MobileNetV2 uses only 2.26M parameters (~11 MB), inference is lightweight (~300 MB total process RAM).
- **Restarting the App:** If the container ever sleeps or runs out of memory, click the bottom right **Manage app** menu -> **Reboot app**.
- **Continuous Deployment:** Any future push to `main` (`git push origin main`) will automatically trigger a rebuild and redeployment on Streamlit Cloud within 1-2 minutes.
