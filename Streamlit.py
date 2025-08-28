# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Set page config
st.set_page_config(page_title="Media Processor", layout="centered")

# Title
st.title("📷 Video/Image Input Processor")
st.write("Upload an image, video, or use your webcam. The output will be a grayscale version.")

# --- File uploader OR camera input ---
input_type = st.radio("Choose input type:", ("Image", "Video", "Camera"))

# Placeholder for processed output
output_placeholder = st.empty()

# Function to process image (example: convert to grayscale)
def process_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Function to process video (convert each frame to grayscale)
def process_video(video_file):
    # Use tempfile to store output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    # OpenCV Video Capture
    cap = cv2.VideoCapture(video_file)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)

    cap.release()
    out.release()
    return temp_output.name

# ---------------------------
# Handle different input types
# ---------------------------

if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read image from buffer
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.subheader("Original Image:")
        st.image(img, channels="BGR")

        # Process image
        processed_img = process_image(img)

        st.subheader("Processed (Grayscale) Image:")
        st.image(processed_img, channels="GRAY")

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        st.subheader("Processing Video...")
        processed_video_path = process_video(tfile.name)

        st.subheader("Processed (Grayscale) Video:")
        with open(processed_video_path, 'rb') as f:
            st.video(f.read())

elif input_type == "Camera":
    captured_image = st.camera_input("Take a photo")

    if captured_image is not None:
        # Read image from camera
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.subheader("Original Captured Image:")
        st.image(img, channels="BGR")

        # Process image
        processed_img = process_image(img)

        st.subheader("Processed (Grayscale) Image:")
        st.image(processed_img, channels="GRAY")
