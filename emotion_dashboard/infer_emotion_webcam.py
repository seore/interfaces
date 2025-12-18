import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

from utils.fer2013 import EMOTIONS

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    x = resized.astype(np.float32) / 255.0
    x = x[None, None, :, :] 
    return x

def ema_smooth(prev, cur, alpha=0.35):
    if prev is None:
        return cur
    return alpha * cur + (1 - alpha) * prev

def main():
    session = ort.InferenceSession("models/emotion.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(0)
    hist = deque(maxlen=120)
    smooth_probs = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        if res.detections:
            # take highest score
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
                # stable softmax
                logits = logits - np.max(logits)
                probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)
                smooth_probs = ema_smooth(smooth_probs, probs, alpha=0.45)

                idx = int(np.argmax(smooth_probs))
                conf = float(smooth_probs[idx])
                label = EMOTIONS[idx]

                hist.append((time.time(), smooth_probs.copy()))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
