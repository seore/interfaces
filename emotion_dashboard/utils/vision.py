import random
import numpy as np
import torch

def to_tensor_grayscale48(img_u8: np.ndarray) -> torch.Tensor:
    x = img_u8.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0)  
    return x

def random_affine_rough(img_u8: np.ndarray) -> np.ndarray:
    # very lightweight augmentation in numpy (no PIL dependency)
    # random shifts + slight noise
    img = img_u8.copy()
    h, w = img.shape

    # shift
    max_shift = 3
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    shifted = np.zeros_like(img)
    xs0 = max(0, dx); xs1 = min(w, w + dx)
    ys0 = max(0, dy); ys1 = min(h, h + dy)
    shifted[ys0:ys1, xs0:xs1] = img[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    img = shifted

    # mild brightness/contrast jitter
    if random.random() < 0.8:
        alpha = 1.0 + random.uniform(-0.15, 0.15)  
        beta = random.uniform(-10, 10)     
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    # random erasing
    if random.random() < 0.3:
        rh = random.randint(6, 14)
        rw = random.randint(6, 14)
        y = random.randint(0, h - rh)
        x = random.randint(0, w - rw)
        img[y:y+rh, x:x+rw] = random.randint(0, 255)

    return img

class FERTransform:
    def __init__(self, train: bool):
        self.train = train

    def __call__(self, img_u8):
        if self.train:
            img_u8 = random_affine_rough(img_u8)
        return to_tensor_grayscale48(img_u8)
