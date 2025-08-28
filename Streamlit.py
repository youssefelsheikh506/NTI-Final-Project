import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import torch
from torchvision import transforms
from PIL import Image

# ------------------------
# 🔹 Load your PyTorch model
# ------------------------

@st.cache_resource
def load_model():
# Load your model correctly
    model = YOLO("transfer_yolo.pt")
    model.eval()
    return model

model = load_model()

# ------------------------
# 🔹 Preprocessing function
# ------------------------

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Assuming RGB input
])

# ------------------------
# 🔹 Model inference function
# ------------------------

def run_model_on_image(img):
    """
    Takes an OpenCV BGR image, runs YOLOv8 model inference using Ultralytics,
    and returns the image with bounding boxes drawn.
    """
    # Run YOLO inference (Ultralytics handles RGB conversion internally)
    results = model(img)
    
    # Get the first result and draw predictions on the image
    annotated_img = results[0].plot()  # OpenCV BGR image with boxes/labels
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return annotated_img


# ------------------------
# 🔹 Streamlit UI
# ------------------------

st.set_page_config(page_title="Model-Powered Media Processor", layout="centered")
st.title("🧠 Model Inference on Images and Video")
st.write("Upload an image, video, or use your webcam. The model will process it and return output.")

input_type = st.radio("Choose input type:", ("Image", "Video", "Camera"))

# ------------------------
# 🔹 IMAGE INPUT
# ------------------------

if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.subheader("Original Image:")
        st.image(img, channels="BGR")

        st.subheader("Model Output:")
        result = run_model_on_image(img)

        if isinstance(result, str):
            st.success(result)
        else:
            st.image(result, channels="BGR")

# ------------------------
# 🔹 VIDEO INPUT
# ------------------------

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        st.subheader("Original Video:")
        st.video(tfile.name)

        st.subheader("Processing Video...")

        # Temporary output path
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        cap = cv2.VideoCapture(tfile.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        count = 0
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # Run model
            result = run_model_on_image(frame)

            if isinstance(result, np.ndarray):
                processed_frame = cv2.resize(result, (width, height))
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            else:
                # Fallback: just write original frame
                processed_frame = frame

            out.write(processed_frame)

            count += 1
            progress.progress(min(count / frame_count, 1.0))

        cap.release()
        out.release()

        st.success("Video processed!")
        st.subheader("Processed Video:")
        st.video(temp_output.name)

# ------------------------
# 🔹 CAMERA INPUT
# ------------------------

elif input_type == "Camera":
    captured_image = st.camera_input("Take a photo")

    if captured_image is not None:
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.subheader("Captured Image:")
        st.image(img, channels="BGR")

        st.subheader("Model Output:")
        result = run_model_on_image(img)

        if isinstance(result, str):
            st.success(result)
        else:
            st.image(result, channels="RGB")
