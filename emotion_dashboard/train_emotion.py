import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from models.emotion_cnn import EmotionCNN
from utils.fer2013 import FER2013Dataset, EMOTIONS
from utils.vision import FERTransform

def seed_all(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_weighted_sampler(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights_per_class = 1.0 / np.clip(counts, 1, None)
    weights = weights_per_class[labels]
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(labels), replacement=True)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    acc = (ys == ps).mean()
    return loss_sum / len(loader.dataset), acc, ys, ps

def main():
    seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_path = os.path.join("data", "fer2013.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Place FER2013 CSV in emotion_dashboard/data/fer2013.csv")

    train_ds = FER2013Dataset(csv_path, usage="Training", transform=FERTransform(train=True))
    val_ds   = FER2013Dataset(csv_path, usage="PublicTest", transform=FERTransform(train=False))

    sampler = make_weighted_sampler(train_ds.y, num_classes=len(EMOTIONS))
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_acc = -1.0
    os.makedirs("models", exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc, ys, ps = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "emotions": EMOTIONS}, "models/emotion_best.pt")
            print("  -> saved models/emotion_best.pt")

    # Report
    ckpt = torch.load("models/emotion_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    val_loss, val_acc, ys, ps = evaluate(model, val_loader, device)
    print("\nBest val acc:", val_acc)

    report = classification_report(ys, ps, target_names=EMOTIONS, digits=4)
    print(report)

    cm = confusion_matrix(ys, ps)
    os.makedirs("models/reports", exist_ok=True)
    with open("models/reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (PublicTest)")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/reports/confusion_matrix.png", dpi=180)

    with open("models/reports/history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
