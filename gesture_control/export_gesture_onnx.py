import os
import torch
from models.gesture_mlp import GestureMLP

def main():
    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/gesture_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Train first: python train_gesture_mlp.py")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    labels = ckpt["labels"]
    model = GestureMLP(in_dim=63, num_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 63)
    out_path = "models/gesture.onnx"
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("Exported:", out_path)

if __name__ == "__main__":
    main()
