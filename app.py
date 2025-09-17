# app.py
import streamlit as st
from pathlib import Path
import tempfile
import shutil
import time
import pandas as pd

# Use your existing backend functions
from main import predict_sensor, predict_vision
from src.utils.paths import normalize_path  # kept for completeness even if not used directly

# ----------------------------- Page config -----------------------------
st.set_page_config(page_title="Robotic Welding Maintenance (v3)", layout="wide")

# ----------------------------- Styles -----------------------------
CARD_OK = "background:#e7f9ed;border:1px solid #b7f3cd;border-radius:12px;padding:14px"
CARD_WARN = "background:#fff4e6;border:1px solid #ffd8a8;border-radius:12px;padding:14px"
CARD_NEUTRAL = "background:#f1f3f5;border:1px solid #dee2e6;border-radius:12px;padding:14px"

st.markdown(
    """
    <style>
    .metric-card h3 { margin: 0 0 10px 0; font-size: 1.05rem; }
    .metric-card p { margin: 0; font-size: 0.95rem; }
    .subtle { color:#666; font-size: 0.85rem; }
    .footer-note { color:#888; font-size: 0.8rem; margin-top: 1rem; }
    .section-divider { margin: 0.5rem 0 1.25rem 0; border-top: 1px solid #eee; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Header -----------------------------
st.title("🤖 Predictive Maintenance — Robotic Welding Arm (v3)")
st.write("Upload a sensor CSV and a welding image to run the full pipeline. All steps are explicit and easy to follow.")

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Sensor model", ["rf", "lstm"])
sensor_csv_file = st.sidebar.file_uploader("Upload sensor CSV", type=["csv"])
image_file = st.sidebar.file_uploader("Upload welding image", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
conf_thres = st.sidebar.slider("Vision confidence threshold", 0.05, 0.90, 0.25, 0.05)
iou_thres = st.sidebar.slider("Vision IoU threshold", 0.10, 0.90, 0.45, 0.05)
clear_outputs = st.sidebar.checkbox("Clear YOLO outputs before run", value=True)
force_fail = st.sidebar.checkbox("Force sensor fail (testing)", value=False)

st.sidebar.markdown(
    "<p class='subtle'>Tip: Clear outputs to avoid mixing results from previous runs.</p>",
    unsafe_allow_html=True,
)

# ----------------------------- Helpers -----------------------------
def _persist_upload_to_tmp(upload, suffix: str) -> Path:
    """
    Persist an uploaded file to a temporary directory so existing CLI-style
    functions can read them from disk. Returns the path to the temp file.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    out = tmp_dir / f"upload{suffix}"
    with open(out, "wb") as f:
        f.write(upload.getvalue())
    return out

def _find_latest_vision_outputs(project_root: Path) -> tuple[Path | None, Path | None]:
    """
    Return the most recent annotated image and its labels directory (if any),
    assuming YOLO writes to docs/vision_preds.
    """
    predictions_dir = project_root / "docs" / "vision_preds"
    labels_dir = predictions_dir / "labels"
    if not predictions_dir.exists():
        return None, None
    imgs = sorted(
        [p for p in predictions_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".png"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return (imgs[0] if imgs else None, (labels_dir if labels_dir.exists() else None))

def _display_verdict(sensor_pred: int, sensor_conf: float, seams: int):
    """
    Render a verdict card based on the combined results:
    - Healthy: sensor_pred == 0 and seams > 0
    - Maintenance: otherwise
    """
    if sensor_pred == 0 and seams > 0:
        verdict = "✅ System Healthy"
        style = CARD_OK
    else:
        verdict = "⚠️ Maintenance Required"
        style = CARD_WARN
    st.markdown(
        f"<div class='metric-card' style='{style}'>"
        f"<h3>{verdict}</h3>"
        f"<p><b>Sensor:</b> {sensor_pred} @ {sensor_conf:.2f} &nbsp; | &nbsp; "
        f"<b>Seams:</b> {seams}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _display_input_cards(sensor_uploaded: bool, image_uploaded: bool):
    """
    Show quick input status cards side-by-side for clarity.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='metric-card' style='{CARD_NEUTRAL}'>"
            f"<h3>Sensor input</h3>"
            f"<p>{'CSV uploaded' if sensor_uploaded else 'No CSV uploaded'}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card' style='{CARD_NEUTRAL}'>"
            f"<h3>Vision input</h3>"
            f"<p>{'Image uploaded' if image_uploaded else 'No image uploaded'}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

def _display_csv_preview(upload):
    """
    Show a preview of the CSV and a small line chart of the first few numeric columns.
    """
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return
    st.markdown("**Preview sensor CSV**")
    st.dataframe(df.head(20), use_container_width=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.markdown("<p class='subtle'>Quick look at up to 3 numeric columns.</p>", unsafe_allow_html=True)
        st.line_chart(df[numeric_cols[: min(3, len(numeric_cols))]])

# ----------------------------- Tabs -----------------------------
tab_pipeline, tab_sensor, tab_vision = st.tabs(["Full pipeline", "Sensor only", "Vision only"])

# =================================================================
# Full pipeline tab
# =================================================================
with tab_pipeline:
    st.subheader("Run full pipeline")
    st.write("This runs the sensor model on your CSV and YOLO on your image, then combines both into a final verdict.")
    _display_input_cards(sensor_uploaded=bool(sensor_csv_file), image_uploaded=bool(image_file))

    # Run button (full pipeline)
    run_pipeline_btn = st.button("Run full pipeline", type="primary", use_container_width=True,
                                 help="Runs sensor + vision + verdict")
    if run_pipeline_btn:
        if not sensor_csv_file or not image_file:
            st.warning("Please upload both a sensor CSV and an image.", icon="⚠️")
        else:
            # Persist uploads to disk so your existing functions can read them
            csv_path = _persist_upload_to_tmp(sensor_csv_file, ".csv")
            img_path = _persist_upload_to_tmp(image_file, Path(image_file.name).suffix)

            # Build args namespace for your backend
            class Args:
                def __init__(self):
                    self.model = model_choice
                    self.sensor_csv = str(csv_path)
                    self.image = str(img_path)
                    self.force_fail = force_fail
                    self.conf = conf_thres
                    self.iou = iou_thres
                    self.clear_outputs = clear_outputs

            args = Args()

            # Run sensor prediction
            with st.spinner("Running sensor model..."):
                sensor_pred, sensor_conf = predict_sensor(args)
                time.sleep(0.05)

            # Run vision prediction
            with st.spinner("Running vision model..."):
                seams_detected = predict_vision(args)
                time.sleep(0.05)

            # Display combined verdict
            _display_verdict(sensor_pred, sensor_conf, seams_detected)

            # Show annotated output + labels count if present
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("#### Vision outputs")
            proj_root = Path.cwd()
            annotated_img, labels_dir = _find_latest_vision_outputs(proj_root)
            colA, colB = st.columns([2, 1])
            with colA:
                if annotated_img and annotated_img.exists():
                    st.image(str(annotated_img), caption="YOLO annotated output", use_column_width=True)
                else:
                    st.info("No annotated output image found under docs/vision_preds.")
            with colB:
                if labels_dir and labels_dir.exists():
                    num_labels = len(list(labels_dir.glob("*.txt")))
                    st.markdown(
                        f"<div class='metric-card' style='{CARD_NEUTRAL}'>"
                        f"<h3>Vision summary</h3>"
                        f"<p><b>Label files:</b> {num_labels}</p>"
                        f"<p class='subtle'>Saved under docs/vision_preds/labels</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No label files found under docs/vision_preds/labels.")

            st.markdown("<p class='footer-note'>Outputs are written to docs/vision_preds.</p>", unsafe_allow_html=True)

# =================================================================
# Sensor-only tab
# =================================================================
with tab_sensor:
    st.subheader("Sensor prediction")
    st.write("Upload a sensor CSV and select a model to get the prediction and confidence.")

    # Run button (sensor only)
    run_sensor_btn = st.button("Run sensor only", use_container_width=True)
    if run_sensor_btn:
        if not sensor_csv_file:
            st.warning("Please upload a sensor CSV.", icon="⚠️")
        else:
            csv_path = _persist_upload_to_tmp(sensor_csv_file, ".csv")

            # Build args for sensor
            class Args:
                def __init__(self):
                    self.model = model_choice
                    self.sensor_csv = str(csv_path)
                    self.force_fail = force_fail

            args = Args()

            with st.spinner("Running sensor model..."):
                s_pred, s_conf = predict_sensor(args)
                time.sleep(0.05)

            st.markdown(
                f"<div class='metric-card' style='{CARD_NEUTRAL}'>"
                f"<h3>Sensor result</h3>"
                f"<p><b>Prediction:</b> {s_pred} &nbsp; | &nbsp; <b>Confidence:</b> {s_conf:.3f}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # CSV preview and chart (if uploaded)
    if sensor_csv_file:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        _display_csv_preview(sensor_csv_file)

# =================================================================
# Vision-only tab
# =================================================================
with tab_vision:
    st.subheader("Vision prediction")
    st.write("Upload a welding image and run YOLO seam detection.")

    # Run button (vision only)
    run_vision_btn = st.button("Run vision only", use_container_width=True)
    if run_vision_btn:
        if not image_file:
            st.warning("Please upload an image.", icon="⚠️")
        else:
            img_path = _persist_upload_to_tmp(image_file, Path(image_file.name).suffix)

            # Build args for vision
            class Args:
                def __init__(self):
                    self.image = str(img_path)
                    self.conf = conf_thres
                    self.iou = iou_thres
                    self.clear_outputs = clear_outputs

            args = Args()

            with st.spinner("Running YOLO..."):
                seams = predict_vision(args)
                time.sleep(0.05)

            # Verdict here uses a neutral sensor baseline
            _display_verdict(sensor_pred=0, sensor_conf=1.0, seams=seams)

            # Show annotated image if present
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            proj_root = Path.cwd()
            annotated_img, labels_dir = _find_latest_vision_outputs(proj_root)
            if annotated_img and annotated_img.exists():
                st.image(str(annotated_img), caption="YOLO annotated output", use_column_width=True)
            else:
                st.info("No annotated output image found under docs/vision_preds.")

            if labels_dir and labels_dir.exists():
                num_labels = len(list(labels_dir.glob("*.txt")))
                st.markdown(
                    f"<div class='metric-card' style='{CARD_NEUTRAL}'>"
                    f"<h3>Vision summary</h3>"
                    f"<p><b>Label files:</b> {num_labels}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No label files found under docs/vision_preds/labels.")

# ----------------------------- Footer -----------------------------
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<p class='footer-note'>Session settings: model = "
    f"<b>{model_choice}</b>, conf = <b>{conf_thres:.2f}</b>, iou = <b>{iou_thres:.2f}</b>, "
    f"clear_outputs = <b>{clear_outputs}</b>, force_fail = <b>{force_fail}</b>.</p>",
    unsafe_allow_html=True,
)
