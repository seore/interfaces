import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from utils.fer2013 import EMOTIONS

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return x[None, None, :, :]  # (1,1,48,48)

def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / (np.sum(e) + 1e-9)

def ema_smooth(prev, cur, alpha=0.45):
    return cur if prev is None else (alpha * cur + (1 - alpha) * prev)

def pick_largest_face(faces):
    # faces: list of (x,y,w,h)
    return max(faces, key=lambda b: b[2] * b[3])

def expand_box(x, y, w, h, img_w, img_h, scale=1.2):
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
    # Load ONNX
    session = ort.InferenceSession("models/emotion.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # OpenCV Haar cascade (bundled with opencv-python)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing camera index (0/1).")

    hist = deque(maxlen=120)
    smooth_probs = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_full,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) > 0:
            x, y, fw, fh = pick_largest_face(faces)
            x1, y1, x2, y2 = expand_box(x, y, fw, fh, w, h, scale=1.15)
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                inp = preprocess_face(face)
                logits = session.run(None, {input_name: inp})[0][0]
                probs = softmax(logits)
                smooth_probs = ema_smooth(smooth_probs, probs, alpha=0.45)

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

        cv2.imshow("Emotion (OpenCV face detect) - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
