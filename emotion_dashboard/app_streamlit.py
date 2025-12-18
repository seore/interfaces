import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import streamlit as st

from utils.fer2013 import EMOTIONS

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    x = resized.astype(np.float32) / 255.0
    return x[None, None, :, :]

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def ema(prev, cur, alpha=0.45):
    return cur if prev is None else alpha * cur + (1 - alpha) * prev

def main():
    st.set_page_config(page_title="Emotion Dashboard", layout="wide")
    st.title("Face Emotion Recognition Dashboard")

    col1, col2 = st.columns([2, 1])

    with col2:
        cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
        run = st.toggle("Run", value=False)
        st.caption("Tip: If camera doesn't show, try index 1 on macOS.")

    session = ort.InferenceSession("models/emotion.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(int(cam_index))
    hist = deque(maxlen=180)  # ~6 sec at 30fps
    smooth_probs = None

    frame_slot = col1.empty()
    metric_slot = col2.empty()
    chart_slot = col2.empty()

    last_draw = time.time()

    while run:
        ok, frame = cap.read()
        if not ok:
            st.error("Camera read failed. Try another camera index.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        label = "—"
        conf = 0.0

        if res.detections:
            det = sorted(res.detections, key=lambda d: d.score[0], reverse=True)[0]
            box = det.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                x = preprocess_face(face)
                logits = session.run(None, {input_name: x})[0][0]
                probs = softmax(logits)
                smooth_probs = ema(smooth_probs, probs, alpha=0.45)

                idx = int(np.argmax(smooth_probs))
                conf = float(smooth_probs[idx])
                label = EMOTIONS[idx]
                hist.append((time.time(), smooth_probs.copy()))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

        metric_slot.metric("Current emotion", label, f"{conf:.2f}")

        # update chart ~5fps to keep streamlit smooth
        if hist and (time.time() - last_draw) > 0.2:
            last_draw = time.time()
            latest = hist[-1][1]
            chart_slot.bar_chart({EMOTIONS[i]: float(latest[i]) for i in range(len(EMOTIONS))})

    cap.release()

if __name__ == "__main__":
    main()
