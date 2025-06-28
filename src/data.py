import os 
import torch 
from PIL import Image
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ClfDataset(Dataset): 
    def __init__(self, path): 
        super().__init__()
        self.path = path 
        self.images = [x for x in os.listdir(self.path) if x.endswith('.png')]
        self.labels = pd.read_csv(f'{path}/labels.csv').set_index('Image')
    def __len__(self): 
        return len(self.images)
    def __getitem__(self,idx):  
        img = Image.open(os.path.join(self.path, self.images[idx])).convert('RGB').resize((226,226))
        img = img.crop((128,38,200,158))
        img_tensor = torch.tensor(np.array(img)).permute(2,0,1) / 255.0
        label = self.labels.loc[self.images[idx]]['Label']
        return img_tensor, torch.tensor(label)

if __name__ == '__main__': 
    data = ClfDataset('data',8) 
    dataloder = DataLoader(data, batch_size=32, shuffle=True, prefetch_factor=2, num_workers=4)
    print(next(iter(dataloder)))

