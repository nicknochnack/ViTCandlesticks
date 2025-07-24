# Used for path manipulation
import os
# Convert label to array
import torch
# Used to load images 
from PIL import Image
# Used to open csv labels file
import pandas as pd
# Used to convert PIL image to array
import numpy as np
# Standard torch dataset and dataloader classes
from torch.utils.data import Dataset, DataLoader
# Albumentations lib for data transformations, pretty vanilla ones used
import albumentations as A
# Output sample transformed images - mainly used for testing 
from matplotlib import pyplot as plt
# Einops for patch testing 
from einops import rearrange 
# Image grid for plotting out patch examples
from mpl_toolkits.axes_grid1 import ImageGrid

class ClfDataset(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        # Set path and train flag
        self.path = path
        # This will be used to drive image augmentations
        self.train = train
        # Exclude zeros - I originally captured non-pattern examples to have a nothing class, dropped it
        self.image_df = pd.read_csv(f"{self.path}/labels.csv")
        self.image_df = self.image_df[self.image_df['Label'] != 0]
        self.image_df['Label'] = self.image_df['Label'] - 1

        # Get all the images from the labels file 
        self.images = list(self.image_df['Image'].values)
        # Get all the labels 
        self.labels = self.image_df.set_index("Image")
        # Build the albumentations image augmentation pipeline, not Color Jitter is only applied if we're training
        self.transform = A.Compose(
            [
                A.Resize(224,224),
                *([A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4)] if train else []),
                A.Crop(x_min=130, y_min=43, x_max=202, y_max=147),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.ToTensorV2(),
            ]
        )

    # Mandatory for Pytorch Datasets - used to determine how many images in a full epoch
    def __len__(self):
        # I originally tested a subset of images when prototyping, 500 for train and 100 for test. This meant faster training runs but wasn't intended to scale. 
        return len(self.images)

    # Get a single image and label pair from the dataset
    def __getitem__(self, idx):
        # Get image name and subfolder (which I've called type) 
        image_name = self.images[idx]
        image_type = self.labels.loc[image_name]['Type']

        # Open the image with Pillow and convert it to RGB...instead of RGBA aka drop the alpha channel from a png file. 
        img = Image.open(os.path.join(self.path, image_type, image_name)).convert("RGB")
        # Apply transforms from albumentations
        img_tensor = self.transform(image=np.array(img))
        # Grab le label 
        label = self.labels.loc[self.images[idx]]["Label"]
        # Return those bad boys to the pipeline  
        return img_tensor["image"], torch.tensor(label)

# TEST TEST TEST
if __name__ == "__main__":
    # Create a sample dataset
    data = ClfDataset("data/test_data")
    # Pass to the dataloader for batching, note prefetch_factor and num_workers throw errors on mac for some reason, works fine in Linux env. 
    dataloder = DataLoader(
        data, batch_size=32, shuffle=True, #prefetch_factor=2, num_workers=4
    )
    # Go through the back and output the transformed images 
    X, y = next(iter(dataloder))
    print(X.shape, y.shape)

    # Add this in for the YouTube video to show the cropping results 
    output_samples = False 
    output_patches = False 
    if output_samples == True:
        for idx, img in enumerate(X):
            img_np = img.permute(1, 2, 0).numpy()
            img_min, img_max = img_np.min(), img_np.max()
            img_np_scaled = (img_np - img_min) / (img_max - img_min)
            plt.imsave(f"images/transformed_images/{idx}_scaled.png", img_np_scaled)
    
    # Add this in for the YouTube video to show outputs after applying the image patch transform - pull viz from archive 
    if output_patches == True: 
        patch_size = 8
        batch_size, channels, height, width = X.shape

        # Double check that we've got appropriate dimensions to compute patches 
        assert height % patch_size == 0, "Image height needs to be divisible by patch size" 
        assert width % patch_size == 0, "Image width needs to be divisble by patch size" 
        # Main patching function 
        res = rearrange(X, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        for idx, img in enumerate(res):
            reconstructed_image = rearrange(img, "s (p1 p2 c) -> s p1 p2 c", p1=patch_size, p2=patch_size)
            fig = plt.figure(figsize=(10., 10.))
            grid = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(height // patch_size, width // patch_size),  
                            axes_pad=0.1,  # pad between Axes in inch.
                            )

            for idx, ax in enumerate(grid):
                ax.imshow(reconstructed_image[idx])
            plt.show()
  
        
        