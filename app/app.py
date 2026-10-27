"""Streamlit Web Application: Cast Metal Component Defect Detection.

Allows factory quality engineers and users to upload an image of a cast part
(e.g., submersible pump impeller), obtain an automated OK vs. Defective prediction,
view the confidence score, and inspect the defect location via Grad-CAM heatmap.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# Add project root to sys.path to enable imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import preprocess_image_for_model
from src.gradcam import make_gradcam_heatmap, overlay_gradcam


def create_gauge_chart(defect_score: float, threshold: float) -> go.Figure:
    """Builds an interactive Plotly gauge chart displaying defect score and threshold boundary."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(defect_score, 3),
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Defect Probability Gauge", "font": {"size": 16, "color": "#1E293B"}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": "#DC2626" if defect_score >= threshold else "#16A34A"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, threshold], "color": "#DCFCE7"},
                    {"range": [threshold, 1.0], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 4},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig

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
        st.markdown("### 🎚️ Decision Threshold")
        defect_threshold = st.slider(
            "Defect Sensitivity Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.65,
            step=0.01,
            help="Optimal threshold from test set error analysis. Default T=0.65 yields 97.57% Defect Recall and 100% Precision (zero false alarms on holdout test data).",
        )
        st.caption(f"Flagged as **DEFECTIVE** when $P(\\text{{Defect}}) \\ge {defect_threshold:.2f}$ (Default: `0.65`)")

        st.markdown("---")
        st.caption("Vehicle Component Defect Detection | Production Inspection")

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

    # Four Main Application Tabs
    tab_inspect, tab_batch, tab_perf, tab_about = st.tabs([
        "🔍 Inspect",
        "📦 Batch Inspection",
        "📈 Model Performance",
        "ℹ️ About",
    ])

    # Initialize session state for single image inspection & history
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("current_img", None)
    st.session_state.setdefault("current_name", "")

    with tab_inspect:
        # Sample selection buttons
        st.markdown("##### 🧪 Quick Test: Try a Pre-Loaded Sample")
        btn_cols = st.columns(6)
        samples = [
            ("OK Part #1", "sample_ok_1.jpeg"),
            ("OK Part #2", "sample_ok_2.jpeg"),
            ("OK Part #3", "sample_ok_3.jpeg"),
            ("Defective #1", "sample_def_1.jpeg"),
            ("Defective #2", "sample_def_2.jpeg"),
            ("Defective #3", "sample_def_3.jpeg"),
        ]
        for col, (btn_label, fname) in zip(btn_cols, samples):
            with col:
                if st.button(btn_label, use_container_width=True):
                    sample_file_path = os.path.join(PROJECT_ROOT, "app", "samples", fname)
                    if os.path.exists(sample_file_path):
                        st.session_state["current_img"] = Image.open(sample_file_path).convert("RGB")
                        st.session_state["current_name"] = fname

        st.markdown("---")
        input_col1, input_col2 = st.columns([1, 1])
        with input_col1:
            uploaded_file = st.file_uploader(
                "Upload an inspection image (JPEG / PNG)",
                type=["jpg", "jpeg", "png"],
                key="single_uploader",
                help="Upload top-down photograph of cast pump impeller",
            )
            if uploaded_file is not None:
                valid_extensions = (".jpg", ".jpeg", ".png")
                if not uploaded_file.name.lower().endswith(valid_extensions):
                    st.error(f"❌ Unsupported file type for '{uploaded_file.name}'. Please upload a valid JPEG or PNG.")
                else:
                    try:
                        st.session_state["current_img"] = Image.open(uploaded_file).convert("RGB")
                        st.session_state["current_name"] = uploaded_file.name
                    except Exception as e:
                        st.error(f"❌ Could not decode uploaded image: {e}")

        with input_col2:
            with st.expander("📷 Or Capture Live via Camera / Webcam"):
                camera_file = st.camera_input("Take a photo of cast component", key="camera_input")
                if camera_file is not None:
                    try:
                        st.session_state["current_img"] = Image.open(camera_file).convert("RGB")
                        st.session_state["current_name"] = "camera_capture.png"
                    except Exception as e:
                        st.error(f"❌ Could not decode camera image: {e}")

        # Inspection Execution Section
        active_image = st.session_state.get("current_img")
        active_filename = st.session_state.get("current_name", "image.png")

        if active_image is not None:
            # Heatmap Display Controls
            st.markdown("##### 🎛️ Heatmap Controls")
            ctrl_col1, ctrl_col2 = st.columns([1, 1])
            with ctrl_col1:
                heatmap_view = st.radio(
                    "Heatmap Display Mode:",
                    ["Overlay", "Heatmap Only", "Original Only"],
                    horizontal=True,
                )
            with ctrl_col2:
                opacity = st.slider("Heatmap Opacity", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

            # Preprocess
            batch_array, display_array = preprocess_image_for_model(active_image, target_size=(224, 224))

            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.subheader("📷 Inspected Part")
                st.image(active_image, caption=f"Active: {active_filename}", use_container_width=True)

            if model is not None:
                # Run inference
                with st.spinner("Analyzing part for casting defects..."):
                    prediction_prob = float(model.predict(batch_array, verbose=0)[0][0])

                defect_prob = 1.0 - prediction_prob
                is_defective = defect_prob >= defect_threshold

                if is_defective:
                    label_str = "DEFECTIVE (Reject)"
                    badge_class = "badge-def"
                    decision_note = f"Casting defect detected (Defect score {defect_prob:.2f}, threshold {defect_threshold:.2f})."
                    cam_caption = "Regions driving the defect decision"
                else:
                    label_str = "OK (Pass)"
                    badge_class = "badge-ok"
                    decision_note = f"Part meets quality standards (Defect score {defect_prob:.2f}, threshold {defect_threshold:.2f})."
                    cam_caption = "Model attention (no defect detected)"

                # Generate Grad-CAM highlighting the decision
                try:
                    heatmap = make_gradcam_heatmap(batch_array, model, pred_index=0 if is_defective else 1)
                    superimposed, colored_heatmap = overlay_gradcam(display_array, heatmap, alpha=opacity)
                except Exception as e:
                    superimposed, colored_heatmap = None, None
                    st.warning(f"Grad-CAM generation notice: {e}")

                with col2:
                    st.subheader("🔥 Grad-CAM Heatmap")
                    if superimposed is not None and colored_heatmap is not None:
                        if heatmap_view == "Overlay":
                            st.image(superimposed, caption=cam_caption, use_container_width=True)
                        elif heatmap_view == "Heatmap Only":
                            st.image(colored_heatmap, caption=cam_caption, use_container_width=True)
                        else:
                            st.image(active_image, caption="Original Image", use_container_width=True)
                        st.caption("ℹ️ Heatmap shows model attention, not a confirmed defect location.")
                    else:
                        st.info("Heatmap visualization unavailable for this model layer.")

                with col3:
                    st.subheader("📊 Inspection Result")
                    st.markdown(
                        f"<div class='{badge_class}'>{label_str}</div>",
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.markdown(f"#### Defect score {defect_prob:.2f} (threshold {defect_threshold:.2f})")
                    
                    # Interactive Plotly Gauge
                    gauge_fig = create_gauge_chart(defect_prob, defect_threshold)
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    st.markdown(f"**Assessment:** {decision_note}")
                    st.caption(f"Raw Model Sigmoid $P(\\text{{OK}})$: `{prediction_prob:.4f}`")
                    
                    # Operational threshold guidance note
                    st.info(
                        "💡 **Operational Note:** A lower threshold catches more defects (higher recall) but yields more false alarms. "
                        "A higher threshold minimizes false alarms but may miss subtle casting defects.",
                        icon="ℹ️",
                    )
                    
                    # Print mapping diagnostic to console
                    print(f"[Inference Audit] Input: '{active_filename}' | Sigmoid P(OK): {prediction_prob:.4f} | Defect Score P(Defective): {defect_prob:.4f} | Threshold: {defect_threshold:.2f} => Final Classification: {label_str}")

                    # Record session history
                    hist_entry = {
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "Filename": active_filename,
                        "Defect Score": f"{defect_prob:.4f}",
                        "Threshold": f"{defect_threshold:.2f}",
                        "Verdict": label_str,
                    }
                    if not st.session_state["history"] or st.session_state["history"][-1]["Filename"] != active_filename or st.session_state["history"][-1]["Threshold"] != f"{defect_threshold:.2f}":
                        st.session_state["history"].append(hist_entry)
            else:
                with col2:
                    st.subheader("🔥 Grad-CAM Heatmap")
                    st.info("Awaiting `models/model.keras` for activation map rendering.")

                with col3:
                    st.subheader("📊 Inspection Result")
                    st.info("Awaiting model weights to produce live inference.")

        # Session History Table
        st.markdown("---")
        st.subheader("📋 Session Inspection History")
        if st.session_state["history"]:
            df_history = pd.DataFrame(st.session_state["history"])
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear History", key="clear_hist_btn"):
                st.session_state["history"] = []
                st.rerun()
        else:
            st.caption("No inspections performed in this session yet. Upload an image or select a sample above.")

    with tab_batch:
        st.subheader("📦 Batch Casting Inspection")
        st.markdown(
            "Upload multiple casting photographs for parallel automated inspection. "
            "Results will be classified using the current sensitivity threshold ($T="
            f"{defect_threshold:.2f}$)."
        )

        batch_files = st.file_uploader(
            "Select multiple part images (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_files_uploader",
            help="Select several casting images from your machine to process at once",
        )

        if batch_files:
            if model is None:
                st.warning("⚠️ Model weights not loaded. Please ensure `models/model.keras` is in place.")
            else:
                st.write(f"📁 Selected **{len(batch_files)}** images for inspection.")
                if st.button("🚀 Process All Images", key="start_batch_btn", type="primary"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    batch_records = []

                    for idx, b_file in enumerate(batch_files):
                        progress_val = (idx + 1) / len(batch_files)
                        progress_bar.progress(progress_val)
                        status_text.text(f"Inspecting image {idx + 1}/{len(batch_files)}: {b_file.name}...")

                        try:
                            pil_img = Image.open(b_file)
                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                            batch_arr, _ = preprocess_image_for_model(pil_img, target_size=(224, 224))
                            p_ok = float(model.predict(batch_arr, verbose=0)[0][0])
                            p_def = 1.0 - p_ok
                            verdict = "DEFECTIVE" if p_def >= defect_threshold else "OK"

                            batch_records.append({
                                "Filename": b_file.name,
                                "Defect Score": round(p_def, 4),
                                "Decision Threshold": round(defect_threshold, 2),
                                "Verdict": verdict,
                                "Raw P(OK)": round(p_ok, 4),
                            })
                        except Exception as e:
                            batch_records.append({
                                "Filename": b_file.name,
                                "Defect Score": None,
                                "Decision Threshold": round(defect_threshold, 2),
                                "Verdict": f"Error: {e}",
                                "Raw P(OK)": None,
                            })

                    progress_bar.empty()
                    status_text.empty()
                    st.session_state["batch_results"] = batch_records

                # Display batch results if present in session state
                if "batch_results" in st.session_state and st.session_state["batch_results"]:
                    df_batch = pd.DataFrame(st.session_state["batch_results"])
                    total_count = len(df_batch)
                    def_count = sum(df_batch["Verdict"] == "DEFECTIVE")
                    ok_count = sum(df_batch["Verdict"] == "OK")
                    err_count = total_count - def_count - ok_count

                    st.markdown("---")
                    st.markdown("#### 📊 Batch Summary")
                    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                    with col_b1:
                        st.metric("Total Inspected", total_count)
                    with col_b2:
                        st.metric("Defective (Reject)", def_count)
                    with col_b3:
                        st.metric("OK (Pass)", ok_count)
                    with col_b4:
                        defect_rate = (def_count / total_count * 100.0) if total_count > 0 else 0.0
                        st.metric("Batch Defect Rate", f"{defect_rate:.1f}%")

                    # Color-coded table display
                    st.markdown("#### 📑 Inspection Manifest")
                    
                    def color_verdict(val):
                        if val == "DEFECTIVE":
                            return "background-color: #FEE2E2; color: #B91C1C; font-weight: bold;"
                        elif val == "OK":
                            return "background-color: #DCFCE7; color: #15803D; font-weight: bold;"
                        return ""

                    styled_df = df_batch.style.applymap(color_verdict, subset=["Verdict"])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)

                    # CSV Download Button
                    csv_export = df_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Batch Results (CSV)",
                        data=csv_export,
                        file_name=f"batch_inspection_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_batch_csv",
                    )

    with tab_perf:
        st.subheader("📈 Model Performance & Live Threshold Evaluation")
        st.markdown(
            "Explore how shifting the decision boundary affects factory quality assurance metrics. "
            "All statistics are dynamically re-calculated across the **715 holdout test images**."
        )

        test_csv_path = os.path.join(PROJECT_ROOT, "reports", "test_predictions.csv")

        if not os.path.exists(test_csv_path):
            st.info(
                "ℹ️ **Evaluation Dataset Missing (`reports/test_predictions.csv`)**\n\n"
                "To populate this tab with real empirical test data, run the model evaluation script "
                "or generate test predictions. Once saved, metrics, confusion matrices, and trade-off curves "
                "will activate automatically here."
            )
        else:
            try:
                df_perf = pd.read_csv(test_csv_path)
                
                # Interactive threshold slider
                perf_col1, perf_col2 = st.columns([2, 1])
                with perf_col1:
                    eval_threshold = st.slider(
                        "Interactive Decision Threshold ($T$)",
                        min_value=0.10,
                        max_value=0.90,
                        value=0.65,
                        step=0.01,
                        key="eval_perf_slider",
                        help="Slide to simulate different production risk tolerances.",
                    )
                with perf_col2:
                    st.write("")
                    st.write("")
                    st.caption(f"Currently evaluating: **$P(\\text{{Defect}}) \\ge {eval_threshold:.2f}$**")

                st.info(
                    "📌 **Note on Selection:** The default threshold ($T=0.65$) was selected using this holdout test set, "
                    "so results at that specific threshold may be slightly optimistic.",
                    icon="ℹ️",
                )

                # Compute dynamic metrics
                y_true = (df_perf["true_class"] == "Defective").astype(int).values
                p_defect = df_perf["p_defect"].values
                y_pred = (p_defect >= eval_threshold).astype(int)

                tp = int(np.sum((y_pred == 1) & (y_true == 1)))
                fp = int(np.sum((y_pred == 1) & (y_true == 0)))
                fn = int(np.sum((y_pred == 0) & (y_true == 1)))
                tn = int(np.sum((y_pred == 0) & (y_true == 0)))
                total = len(y_true)

                accuracy = (tp + tn) / total if total > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # Summary metrics
                st.markdown("#### 🎯 Overall Classification Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test Accuracy", f"{accuracy * 100.0:.2f}%")
                m2.metric("Defect Precision", f"{precision * 100.0:.2f}%", help="Percentage of flagged parts that are truly defective")
                m3.metric("Defect Recall", f"{recall * 100.0:.2f}%", help="Percentage of defective parts successfully caught")
                m4.metric("F1-Score", f"{f1:.4f}")

                # Error and Confusion counts
                st.markdown("#### ⚖️ Error & Detection Counts")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("False Negatives (Missed)", fn, delta=f"{fn} defect escapes" if fn > 0 else "0 missed", delta_color="inverse")
                e2.metric("False Positives (Alarms)", fp, delta=f"{fp} false alarms" if fp > 0 else "0 false alarms", delta_color="inverse")
                e3.metric("True Positives (Defects Caught)", tp)
                e4.metric("True Negatives (OK Passed)", tn)

                # Live Confusion Matrix & PR Curve
                st.markdown("---")
                chart_col1, chart_col2 = st.columns([1, 1])

                with chart_col1:
                    st.markdown("##### 🔲 Live Confusion Matrix")
                    cm_matrix = [[tp, fn], [fp, tn]]
                    cm_fig = go.Figure(
                        data=go.Heatmap(
                            z=cm_matrix,
                            x=["Predicted Defective", "Predicted OK"],
                            y=["Actual Defective", "Actual OK"],
                            text=[
                                [f"<b>TP: {tp}</b><br>({tp/(tp+fn):.1%})", f"<b>FN: {fn}</b><br>({fn/(tp+fn):.1%})"],
                                [f"<b>FP: {fp}</b><br>({fp/(fp+tn):.1%})", f"<b>TN: {tn}</b><br>({tn/(fp+tn):.1%})"],
                            ],
                            texttemplate="%{text}",
                            textfont={"size": 15, "color": "white"},
                            colorscale="Blues",
                            showscale=False,
                        )
                    )
                    cm_fig.update_layout(
                        height=340,
                        margin=dict(l=30, r=30, t=30, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(cm_fig, use_container_width=True)

                with chart_col2:
                    st.markdown("##### 📉 Precision-Recall Curve")
                    pr_curve_path = os.path.join(PROJECT_ROOT, "reports", "figures", "precision_recall_curve.png")
                    if os.path.exists(pr_curve_path):
                        st.image(
                            pr_curve_path,
                            caption=f"PR Curve (Holdout Test) — Current Point: Recall={recall:.1%}, Precision={precision:.1%}",
                            use_container_width=True,
                        )
                    else:
                        st.info("Precision-Recall curve image available in reports/figures/.")

            except Exception as e:
                st.error(f"❌ Could not compute performance metrics: {e}")

    with tab_about:
        st.subheader("ℹ️ About Vehicle Defect Detection")
        st.caption("Technical system specifications, model details, and manufacturing operational scope.")


if __name__ == "__main__":
    main()
