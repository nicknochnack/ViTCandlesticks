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
        self.train = train
        # Exclude zeros 
        self.image_df = pd.read_csv(f"{self.path}/labels.csv")
        self.image_df = self.image_df[self.image_df['Label'] != 0]
        self.image_df['Label'] = self.image_df['Label'] - 1

        self.images = list(self.image_df['Image'].values)
        self.labels = self.image_df.set_index("Image")
        self.transform = A.Compose(
            [
                A.Resize(224,224),
                *([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4)] if train else []),
                A.Crop(x_min=130, y_min=43, x_max=202, y_max=147),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_type = self.labels.loc[image_name]['Type']

        img = Image.open(os.path.join(self.path, image_type, image_name)).convert("RGB")
        img_tensor = self.transform(image=np.array(img))
        label = self.labels.loc[self.images[idx]]["Label"]
        
        return img_tensor["image"], torch.tensor(label)


if __name__ == "__main__":
    data = ClfDataset("data/test_data")
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