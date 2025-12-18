import numpy as np
import pandas as pd
from torch.utils.data import Dataset

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

class FER2013Dataset(Dataset):
    def __init__(self, csv_path: str, usage: str, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == usage].reset_index(drop=True)

        self.y = df["emotion"].astype(int).to_numpy()
        # pixels are space-separated ints length 2304
        pixels = df["pixels"].str.split().apply(lambda xs: np.array(xs, dtype=np.uint8))
        X = np.stack(pixels.to_numpy(), axis=0).reshape(-1, 48, 48)
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]  
        label = int(self.y[idx])
        if self.transform:
            img = self.transform(img)
        return img, label
