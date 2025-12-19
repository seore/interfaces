import os
import time
import cv2
import numpy as np

from utils.hand_features import segment_hand_ycrcb, largest_contour, contour_features, draw_debug

LABELS = ["fist", "thumbs_up", "open_palm", "pinch"]

def main():
    os.makedirs("data/features_npz", exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera index 1.")

    X, y = [], []
    current_label = None
    recording = False

    print("Hotkeys: 1-4 set label | r start/stop recording | s save | q quit")
    print("Tip: Keep your hand centered, plain background, good lighting.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror
        mask = segment_hand_ycrcb(frame)
        cnt = largest_contour(mask)

        status = f"Label: {current_label} | REC: {recording} | Samples: {len(y)}"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if cnt is not None:
            feat = contour_features(cnt)
            if feat is not None:
                draw_debug(frame, cnt)
                if recording and current_label is not None:
                    X.append(feat)
                    y.append(LABELS.index(current_label))

        # show mask small
        mask_small = cv2.resize(mask, (0,0), fx=0.35, fy=0.35)
        mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[60:60+mask_small.shape[0], 20:20+mask_small.shape[1]] = mask_small

        cv2.imshow("Collect Gesture Features (OpenCV)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
            current_label = LABELS[int(chr(key)) - 1]
            print("Current label:", current_label)
        if key == ord("r"):
            recording = not recording
            print("Recording:", recording)
        if key == ord("s"):
            if len(y) == 0:
                print("Nothing to save yet.")
                continue
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = f"data/features_npz/gesture_features_{ts}.npz"
            np.savez_compressed(
                out,
                X=np.stack(X).astype(np.float32),
                y=np.array(y, dtype=np.int64),
                labels=np.array(LABELS, dtype=object),
            )
            print("Saved:", out)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
