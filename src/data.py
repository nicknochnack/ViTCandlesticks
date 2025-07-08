import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from matplotlib import pyplot as plt


class ClfDataset(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.images = [x for x in os.listdir(self.path) if x.endswith(".png")]
        self.labels = pd.read_csv(f"{path}/labels.csv").set_index("Image")
        self.transform = A.Compose(
            [
                A.Resize(226, 226),
                A.Crop(x_min=128, y_min=38, x_max=200, y_max=158),
                *([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.2)] if train else []),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.images[idx])).convert("RGB")
        img_tensor = self.transform(image=np.array(img))
        label = self.labels.loc[self.images[idx]]["Label"]
        
        return img_tensor["image"], torch.tensor(label)


if __name__ == "__main__":
    data = ClfDataset("zz_generated")
    dataloder = DataLoader(
        data, batch_size=32, shuffle=True, prefetch_factor=2, num_workers=4
    )
    X, y = next(iter(dataloder))
    print(X.shape, y.shape)
    for idx, img in enumerate(X):
        img_np = img.permute(1, 2, 0).numpy()
        img_min, img_max = img_np.min(), img_np.max()
        img_np_scaled = (img_np - img_min) / (img_max - img_min)
        plt.imsave(f"transformed_images/{idx}_scaled.png", img_np_scaled)
