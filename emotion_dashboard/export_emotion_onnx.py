import os
import torch
from models.emotion_cnn import EmotionCNN
from utils.fer2013 import EMOTIONS

def main():
    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/emotion_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Train first: python train_emotion.py")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = EmotionCNN(num_classes=len(EMOTIONS))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 1, 48, 48)
    out_path = "models/emotion.onnx"
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
