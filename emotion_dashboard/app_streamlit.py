import os
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st

from utils.fer2013 import EMOTIONS

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return x[None, None, :, :]

def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / (np.sum(e) + 1e-9)

def ema(prev, cur, alpha=0.45):
    return cur if prev is None else (alpha * cur + (1 - alpha) * prev)

def pick_largest_face(faces):
    return max(faces, key=lambda b: b[2] * b[3])

def expand_box(x, y, w, h, img_w, img_h, scale=1.15):
    cx = x + w / 2
    cy = y + h / 2
    nw = w * scale
    nh = h * scale
    x1 = int(max(0, cx - nw / 2))
    y1 = int(max(0, cy - nh / 2))
    x2 = int(min(img_w, cx + nw / 2))
    y2 = int(min(img_h, cy + nh / 2))
    return x1, y1, x2, y2

def main():
    st.set_page_config(page_title="Emotion Dashboard", layout="wide")
    st.title("Face Emotion Recognition Dashboard (OpenCV detector)")

    col1, col2 = st.columns([2, 1])

    with col2:
        cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
        run = st.toggle("Run", value=False)
        st.caption("If camera is blank on macOS, try Camera index = 1.")

    # Robust model path (works no matter where you launch streamlit from)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "emotion.onnx")

    if not os.path.exists(model_path):
        st.error(f"Missing model: {model_path}\nRun: python3 export_emotion_onnx.py")
        return

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        st.error("Failed to load OpenCV haarcascade_frontalface_default.xml")
        return

    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error("Could not open webcam. Try another camera index.")
        return

    hist = deque(maxlen=180)
    smooth_probs = None

    frame_slot = col1.empty()
    metric_slot = col2.empty()
    chart_slot = col2.empty()

    last_chart = 0.0

    while run:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed. Try another camera index.")
            break

        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_full,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        label = "—"
        conf = 0.0

        if len(faces) > 0:
            x, y, fw, fh = pick_largest_face(faces)
            x1, y1, x2, y2 = expand_box(x, y, fw, fh, w, h, scale=1.15)
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                inp = preprocess_face(face)
                logits = session.run(None, {input_name: inp})[0][0]
                probs = softmax(logits)
                smooth_probs = ema(smooth_probs, probs, alpha=0.45)

                idx = int(np.argmax(smooth_probs))
                conf = float(smooth_probs[idx])
                label = EMOTIONS[idx]
                hist.append((time.time(), smooth_probs.copy()))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)
        metric_slot.metric("Current emotion", label, f"{conf:.2f}")

        if smooth_probs is not None and (time.time() - last_chart) > 0.2:
            last_chart = time.time()
            chart_slot.bar_chart({EMOTIONS[i]: float(smooth_probs[i]) for i in range(len(EMOTIONS))})

    cap.release()

if __name__ == "__main__":
    main()
