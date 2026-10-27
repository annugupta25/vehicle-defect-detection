"""Streamlit Web Application: Cast Metal Component Defect Detection.

Allows factory quality engineers and users to upload an image of a cast part
(e.g., submersible pump impeller), obtain an automated OK vs. Defective prediction,
view the confidence score, and inspect the defect location via Grad-CAM heatmap.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to sys.path to enable imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import preprocess_image_for_model
from src.gradcam import make_gradcam_heatmap, overlay_gradcam

# Page Configuration
st.set_page_config(
    page_title="Cast Metal Defect Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 6px solid #3B82F6;
        background-color: #F8FAFC;
    }
    .badge-ok {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-def {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str):
    """Loads and caches the trained Keras model from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        try:
            import keras
            return keras.models.load_model(model_path)
        except Exception:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None


def main():
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = os.path.join(PROJECT_ROOT, "models", "model.keras")

        st.markdown("**Model File Status:**")
        model_exists = os.path.exists(model_path)
        if model_exists:
            st.success("`models/model.keras` found")
        else:
            st.warning("`models/model.keras` not found")

        st.markdown("---")
        st.markdown("### 📋 Inspection Guide")
        st.markdown(
            """
            - **Defective (`def_front`)**: Surface cracks, pinholes, blowholes, or voids.
            - **OK (`ok_front`)**: Clean, continuous cast impeller surface.
            - **Grad-CAM**: Warmer colors (red/yellow) indicate the spatial region influencing the defect decision.
            """
        )

        st.markdown("---")
        st.caption("Vehicle Component Defect Detection | Phase 1 Setup")

    # Main Header
    st.markdown("<div class='main-title'>🔍 Cast Metal Defect Detection System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Automated visual inspection of cast submersible pump impellers with Grad-CAM explainability.</div>",
        unsafe_allow_html=True,
    )

    # Check for model file
    model = load_trained_model(model_path)

    if model is None:
        st.info(
            """
            ### 📌 Model Weights Not Found (`models/model.keras`)

            **How to obtain the trained model:**
            1. Open `notebooks/02_training.ipynb` in **Google Colab** (GPU runtime).
            2. Run the training pipeline to train on the Kaggle casting dataset.
            3. Download the exported `model.keras` file from Colab.
            4. Place it at: `models/model.keras` in this project.
            5. Refresh this application.

            *Upload functionality is active below for image inspection once the model is placed.*
            """
        )

    # Upload Section
    uploaded_file = st.file_uploader(
        "Upload a cast part inspection image (JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload top-down photograph of cast pump impeller",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

        # Preprocess
        batch_array, display_array = preprocess_image_for_model(image, target_size=(224, 224))

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("📷 Uploaded Part")
            st.image(image, caption="Original Image", use_container_width=True)

        if model is not None:
            # Run inference
            with st.spinner("Analyzing part for casting defects..."):
                prediction_prob = float(model.predict(batch_array, verbose=0)[0][0])

            # Class mapping:
            # By default, image_dataset_from_directory sorts folders alphabetically:
            # Index 0: def_front (Defective)
            # Index 1: ok_front (OK)
            # Therefore, sigmoid close to 1.0 indicates OK; close to 0.0 indicates Defective.
            is_ok = prediction_prob >= 0.5
            if is_ok:
                label_str = "OK (Pass)"
                badge_class = "badge-ok"
                confidence = prediction_prob * 100.0
                decision_note = "No significant casting anomalies detected."
            else:
                label_str = "DEFECTIVE (Reject)"
                badge_class = "badge-def"
                confidence = (1.0 - prediction_prob) * 100.0
                decision_note = "Potential casting defect (crack/porosity/pinhole) identified."

            # Generate Grad-CAM
            try:
                heatmap = make_gradcam_heatmap(batch_array, model)
                superimposed, _ = overlay_gradcam(display_array, heatmap, alpha=0.45)
            except Exception as e:
                superimposed = None
                st.warning(f"Grad-CAM generation notice: {e}")

            with col2:
                st.subheader("🔥 Grad-CAM Heatmap")
                if superimposed is not None:
                    st.image(
                        superimposed,
                        caption="Defect Region Focus",
                        use_container_width=True,
                    )
                else:
                    st.info("Heatmap visualization unavailable for this model layer.")

            with col3:
                st.subheader("📊 Inspection Result")
                st.markdown(
                    f"<div class='{badge_class}'>{label_str}</div>",
                    unsafe_allow_html=True,
                )
                st.write("")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                st.progress(confidence / 100.0)

                st.markdown(f"**Assessment:** {decision_note}")
                st.caption(f"Raw Sigmoid Output: `{prediction_prob:.4f}`")
        else:
            with col2:
                st.subheader("🔥 Grad-CAM Heatmap")
                st.info("Awaiting `models/model.keras` for activation map rendering.")

            with col3:
                st.subheader("📊 Inspection Result")
                st.info("Awaiting model weights to produce live inference.")


if __name__ == "__main__":
    main()
