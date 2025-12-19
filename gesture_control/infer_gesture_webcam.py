import time
from collections import deque

import cv2
import joblib
import numpy as np

from utils.hand_features import segment_hand_ycrcb, largest_contour, contour_features, draw_debug
from utils.actions import gesture_to_action, do_action

class Cooldown:
    def __init__(self, seconds):
        self.seconds = seconds
        self.t = 0.0
    def ready(self):
        return (time.time() - self.t) >= self.seconds
    def hit(self):
        self.t = time.time()

class SwipeDetector:
    def __init__(self, maxlen=10, dx_thresh=80, cooldown_s=0.6):
        self.x_hist = deque(maxlen=maxlen)
        self.dx_thresh = dx_thresh
        self.cooldown = Cooldown(cooldown_s)

    def update(self, cx):
        self.x_hist.append(cx)

    def detect(self):
        if not self.cooldown.ready():
            return None
        if len(self.x_hist) < self.x_hist.maxlen:
            return None
        dx = self.x_hist[-1] - self.x_hist[0]
        if abs(dx) >= self.dx_thresh:
            self.cooldown.hit()
            self.x_hist.clear()
            return "swipe_right" if dx > 0 else "swipe_left"
        return None

def main():
    pack = joblib.load("models/gesture_svm.joblib")
    clf = pack["model"]
    labels = pack["labels"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera index 1.")

    swipe = SwipeDetector(maxlen=10, dx_thresh=80, cooldown_s=0.6)
    static_cd = Cooldown(0.35)

    last_gesture = None
    conf_thresh = 0.70

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        mask = segment_hand_ycrcb(frame)
        cnt = largest_contour(mask)

        gesture = None
        conf = 0.0

        if cnt is not None:
            feat = contour_features(cnt)
            if feat is not None:
                draw_debug(frame, cnt)

                # centroid is inside feature vector (index 8,9)
                cx = feat[8]
                swipe.update(cx)
                sw = swipe.detect()
                if sw:
                    gesture = sw
                    conf = 1.0
                else:
                    probs = clf.predict_proba([feat])[0]
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    gesture = labels[idx]

        if gesture:
            cv2.putText(frame, f"{gesture} {conf:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            # trigger actions
            if gesture.startswith("swipe_"):
                action = gesture_to_action(gesture)
                if action:
                    do_action(action)
            else:
                if conf >= conf_thresh and static_cd.ready():
                    if gesture != last_gesture:
                        action = gesture_to_action(gesture)
                        if action:
                            do_action(action)
                        static_cd.hit()
                    last_gesture = gesture

        # show mask preview
        mask_small = cv2.resize(mask, (0,0), fx=0.35, fy=0.35)
        mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[80:80+mask_small.shape[0], 20:20+mask_small.shape[1]] = mask_small

        cv2.imshow("Gesture Control (OpenCV) - q quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
