# 🧠 Emotion & Posture Detector

A computer vision project that detects a person's **emotion** and evaluates their **sitting posture** (healthy or unhealthy) from an uploaded **image**, **video**, or **live webcam feed** using deep learning and OpenCV.

---

## 🔍 Features

- 🎭 **Emotion Detection**: Classifies facial expressions (e.g. happy, sad, angry, neutral, etc.).
- 💺 **Posture Analysis**: Determines if a person's sitting posture is healthy (upright) or unhealthy (slouching, leaning).
- 📸 Supports:
  - Image Upload (`.jpg`, `.png`)
  - Video Upload (`.mp4`, `.avi`)
  - Real-time Webcam Input

---

## 🧰 Tech Stack

- Python 3.8+
- OpenCV
- TensorFlow / PyTorch
- MediaPipe (for pose estimation)
- Pre-trained Emotion Detection model (e.g., FER2013 or custom CNN)
- Streamlit / Flask (for web interface)

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/emotion-posture-detector.git
cd emotion-posture-detector
