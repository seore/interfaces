import glob
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def load_all_npz():
    paths = sorted(glob.glob("data/features_npz/*.npz"))
    if not paths:
        raise FileNotFoundError("No feature npz found. Run collect_landmarks.py and press 's' to save.")
    Xs, ys = [], []
    labels = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["y"])
        labels = list(d["labels"])
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return X, y, labels

def main():
    os.makedirs("models", exist_ok=True)
    X, y, labels = load_all_npz()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=6.0, kernel="rbf", probability=True))
    ])

    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    print(classification_report(y_val, preds, target_names=labels, digits=4))

    joblib.dump({"model": clf, "labels": labels}, "models/gesture_svm.joblib")
    print("Saved: models/gesture_svm.joblib")

if __name__ == "__main__":
    main()
