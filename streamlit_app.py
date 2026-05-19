import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

from main import AISmartFaceAnalytics, calculate_frame_analytics, draw_advanced_ui, draw_analytics_overlay

st.set_page_config(page_title="AI Smart Face Analytics", layout="wide")

st.title("AI Smart Face Analytics — Streamlit Deployment")
st.write(
    "Upload an image or video file to analyze faces for age, gender, emotion, mask status, lighting quality, "
    "and recognition. The system uses the same AI pipeline as the desktop app, now in a browser interface."
)

MODE_OPTIONS = ["Image Upload", "Video Upload", "Camera Capture"]
mode = st.sidebar.radio("Select input mode", MODE_OPTIONS)

analytics_system = None

@st.cache_resource(show_spinner=False)
def load_system():
    return AISmartFaceAnalytics()

analytics_system = load_system()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def annotate_frame(frame, results):
    for result in results:
        draw_advanced_ui(frame, result)
    analytics = calculate_frame_analytics(results)
    draw_analytics_overlay(frame, analytics)
    return frame


def process_frame(frame):
    results = []
    tracked_faces = analytics_system.face_detector.update_tracking(frame)
    for face_id, bbox in tracked_faces:
        x1, y1, x2, y2 = map(int, bbox)
        pad = 20
        face_img = frame[max(0, y1-pad):min(y2+pad, frame.shape[0]), max(0, x1-pad):min(x2+pad, frame.shape[1])]
        if face_img.size > 0:
            analysis_results = analytics_system.analyze_face(face_img, face_id)
            analysis_results["bbox"] = bbox
            results.append(analysis_results)
            analytics_system.data_logger.log_detection(analysis_results, frame_number=0)
    return results


def show_analytics():
    analytics = analytics_system.data_logger.get_analytics()
    st.subheader("Analytics Summary")
    cols = st.columns(4)
    cols[0].metric("Total Faces", analytics.get("total_faces", 0))
    cols[1].metric("Avg Age", f"{analytics.get('avg_age', 0):.1f}")
    cols[2].metric("Male / Female", f"{analytics.get('gender_counts', {}).get('Male',0)} / {analytics.get('gender_counts', {}).get('Female',0)}")
    cols[3].metric("Mask Detected", sum(analytics.get("mask_counts", {}).values()))

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Gender Distribution")
        if analytics.get("gender_counts"):
            st.bar_chart(analytics["gender_counts"])
        else:
            st.info("No gender data yet.")

    with col2:
        st.write("### Emotion Distribution")
        if analytics.get("emotion_counts"):
            st.bar_chart(analytics["emotion_counts"])
        else:
            st.info("No emotion data yet.")

    col3, col4 = st.columns(2)
    with col3:
        st.write("### Mask Status")
        if analytics.get("mask_counts"):
            st.bar_chart(analytics["mask_counts"])
        else:
            st.info("No mask data yet.")

    with col4:
        st.write("### Lighting Quality")
        if analytics.get("lighting_counts"):
            st.bar_chart(analytics["lighting_counts"])
        else:
            st.info("No lighting data yet.")

    if analytics.get("avg_confidence"):
        st.write("### Average Confidence")
        st.write(analytics["avg_confidence"])


def analyze_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Unable to read the uploaded image.")
        return

    analytics_system.data_logger.reset()
    results = process_frame(image)
    annotated = annotate_frame(image.copy(), results)

    st.image(bgr_to_rgb(annotated), caption="Annotated Image", use_column_width=True)
    if results:
        st.write("### Detected Faces")
        st.table([{"Face ID": r["face_id"], "Name": r["name"], "Gender": r["gender"], "Age Range": r["age_range"], "Emotion": r["emotion"], "Mask": r["mask_status"], "Lighting": r["lighting_quality"]} for r in results])
    else:
        st.warning("No faces detected in the uploaded image.")

    show_analytics()


def analyze_video(uploaded_file, max_frames=50):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Unable to open the uploaded video.")
        return

    analytics_system.data_logger.reset()
    first_annotated = None
    frame_index = 0
    progress_bar = st.progress(0)
    results_summary = []

    while cap.isOpened() and frame_index < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = process_frame(frame)
        if first_annotated is None and results:
            first_annotated = annotate_frame(frame.copy(), results)
        frame_index += 1
        progress_bar.progress(int(frame_index / max_frames * 100))

        if frame_index % 10 == 0:
            st.write(f"Processed {frame_index} frames...")

    cap.release()
    progress_bar.empty()
    os.unlink(tmp_path)

    if first_annotated is not None:
        st.image(bgr_to_rgb(first_annotated), caption="First annotated frame", use_column_width=True)
    else:
        st.warning("No faces detected in the uploaded video frames.")

    show_analytics()


def analyze_camera_image(snapshot):
    file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Unable to read camera capture.")
        return

    analytics_system.data_logger.reset()
    results = process_frame(image)
    annotated = annotate_frame(image.copy(), results)
    st.image(bgr_to_rgb(annotated), caption="Captured Image", use_column_width=True)
    if results:
        st.write("### Detected Faces")
        st.table([{"Face ID": r["face_id"], "Name": r["name"], "Gender": r["gender"], "Age Range": r["age_range"], "Emotion": r["emotion"], "Mask": r["mask_status"], "Lighting": r["lighting_quality"]} for r in results])
    else:
        st.warning("No faces detected in the captured image.")
    show_analytics()


if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        analyze_image(uploaded_file)

elif mode == "Video Upload":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    max_frames = st.sidebar.slider("Max frames to analyze", min_value=10, max_value=200, value=50, step=10)
    if uploaded_file is not None:
        analyze_video(uploaded_file, max_frames=max_frames)

else:
    snapshot = st.camera_input("Capture an image from your camera")
    if snapshot is not None:
        analyze_camera_image(snapshot)

st.sidebar.markdown("---")
st.sidebar.write("## Notes")
st.sidebar.write(
    "- Streamlit analysis runs locally and uses the same face analytics models present in `models/`.")
st.sidebar.write(
    "- For real-time webcam-style analysis in browser, use the camera capture option.")
st.sidebar.write(
    "- If the app is slow, reduce the number of max frames or analyze a single image.")
