import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report
from models.gesture_mlp import GestureMLP

def load_all_npz():
    paths = sorted(glob.glob("data/landmarks_npz/*.npz"))
    if not paths:
        raise FileNotFoundError("No gesture npz found. Run collect_landmarks.py and press 's' to save.")
    Xs, ys = [], []
    labels = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["y"])
        labels = d["labels"].tolist()
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return X, y, labels

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return ys, ps

def main():
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, labels = load_all_npz()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    # 85/15 split
    n_val = int(0.15 * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = GestureMLP(in_dim=63, num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=25)

    best_f1 = -1.0
    best_path = "models/gesture_best.pt"

    for epoch in range(1, 26):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        sched.step()

        ys, ps = eval_model(model, val_loader, device)
        report = classification_report(ys, ps, target_names=labels, digits=4, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        print(f"Epoch {epoch:02d} | val macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"model": model.state_dict(), "labels": labels}, best_path)
            print("  -> saved", best_path)

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    ys, ps = eval_model(model, val_loader, device)
    print("\nBest macro F1:", best_f1)
    print(classification_report(ys, ps, target_names=labels, digits=4))

if __name__ == "__main__":
    main()
