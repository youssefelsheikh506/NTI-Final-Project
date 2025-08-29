import streamlit as st
import tempfile
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ======================================================
# 🔹 Load Models
# ======================================================
@st.cache_resource
def load_all_models():
    fer_model = YOLO("transfer_yolo.pt")              # Emotion detection YOLO
    pose_model = YOLO("yolo11n-pose.pt")   # YOLO11n Pose
    clf_model = load_model("abdo model\Pose_Correction_Model.keras")  # Classifier
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return fer_model, pose_model, clf_model, scaler

fer_model, pose_model, clf_model, scaler = load_all_models()
labels_map = {0: "Healthy", 1: "Unhealthy"}

# ======================================================
# 🔹 Helper Functions
# ======================================================
def extract_keypoints(results, num_points=7):
    """Extract first num_points keypoints from YOLO pose results."""
    if results and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()  # (1, total_points, 2)
        if keypoints.shape[1] >= num_points:
            selected = keypoints[0, :num_points, :]
            return selected.flatten().reshape(1, -1)
    return None

def predict_health_from_pose(frame, pose_results):
    """Predict Healthy/Unhealthy using pose keypoints + classifier."""
    keypoints = extract_keypoints(pose_results, num_points=7)
    if keypoints is not None:
        keypoints_scaled = scaler.transform(keypoints)
        prob = clf_model.predict(keypoints_scaled, verbose=0)[0, 0]
        if prob > 0.5:
            return labels_map[1], prob
        else:
            return labels_map[0], 1 - prob
    return "No person", 0.0

def draw_pose_only(frame, results):
    """Draw ONLY keypoints and skeletons (no detection boxes)."""
    if results and results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()[0]
        for (x, y) in kpts:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    return frame

def run_models(frame):
    """
    Run FER + Pose on frame and return:
    1. The annotated frame with FER + keypoints
    2. The predicted health label (string)
    3. The prediction confidence (float)
    """
    # ---------------- FER ----------------
    fer_results = fer_model(frame, verbose=False)
    frame_with_fer = fer_results[0].plot()  # draw boxes + emotion labels

    # ---------------- Pose ----------------
    pose_results = pose_model(frame, verbose=False)
    annotated = draw_pose_only(frame_with_fer, pose_results)

    # ---------------- Health Classification ----------------
    label, conf = predict_health_from_pose(frame, pose_results)

    return annotated, label, conf



# ======================================================
# 🔹 Streamlit UI
# ======================================================
st.set_page_config(page_title="FER + Pose + Health", layout="centered")
st.title("🧠 Emotion + Pose + Health Classification")

option = st.radio("Choose input source:", ("Upload Image", "Upload Video", "Use Webcam"))

# ------------------ IMAGE ------------------
if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.subheader("Original Image")
        st.image(img, channels="BGR")

        annotated, label, conf = run_models(img)

        st.subheader("FER + Pose Result:")
        st.image(annotated, channels="BGR")
        st.markdown(f"### 🏥 Health Prediction: **{label}**")

# ------------------ VIDEO ------------------
elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        text_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run both models (FER + Pose + Health classifier)
            annotated, label, conf = run_models(frame)

            # Convert for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Show prediction in Streamlit (not in frame)
            text_placeholder.markdown(f"### 🏥 Health Prediction: **{label}**")

        cap.release()


# ------------------ WEBCAM ------------------
elif option == "Use Webcam":
    st.warning("Click 'Stop' to end the webcam stream.")
    run = st.checkbox("Start Webcam")
    stframe = st.empty()
    text_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture webcam feed")
            break

        # Run both models (FER + Pose + Health classifier)
        annotated, label, conf = run_models(frame)

        # Convert for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

        # Show prediction in Streamlit (not in frame)
        text_placeholder.markdown(f"### 🏥 Health Prediction: **{label}**")

    cap.release()
