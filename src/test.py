# Bring in model 
from model import ViT
# Grab dataset
from data import ClfDataset
# Bring in dataloader and torch for manipulation
from torch.utils.data import DataLoader
import torch
# Matplotlib for outputting results chart 
from matplotlib import pyplot as plt 

# Setup labels dict
candles = {
    0:"doji",
    1:"bullish_engulfing",
    2:"bearish_engulfing",
    3:"morning_star",
    4:"evening_star"
}

# Coolio, instantiate the model 
model = ViT()
# Load the checkpoint
model.load_state_dict(torch.load("checkpoints/25_model.pt", weights_only=True, map_location=torch.device('cpu')))
# Set to eval mode - dropout should be disabled. 
model.eval()

# Load up the dataset and batch
data = ClfDataset("data/test_data", train=False)
dataloader = DataLoader(
    data, batch_size=9, shuffle=True
)
# Grab one batch
sample = next(iter(dataloader))
X = sample[0] 
print(X.shape)
# Make some predictions
preds = model(X)

# Apply a softmax for probs
softy = torch.nn.Softmax(dim=1)
print(softy(preds)) 
# Apply argmax to get most likely 
argmaxes = torch.argmax(softy(preds), dim=-1)
print(argmaxes)
# Show what it actually should have been
print(sample[1])

# Calculate loss for funzies. 
loss_fn = torch.nn.CrossEntropyLoss()
print(loss_fn(preds, sample[1]))

# Output all the images to a 4x4 subplot, there's probably too much in the plot tbh 
fig, ax = plt.subplots(3,3) 
axs = ax.flatten()
for act, pred, img, ax in zip(sample[1], argmaxes, X, axs):
    img_np = img.permute(1, 2, 0).numpy()
    img_min, img_max = img_np.min(), img_np.max()
    img_np_scaled = (img_np - img_min) / (img_max - img_min)
    ax.imshow(img_np_scaled)
    ax.set_title(f'Actual:{candles[int(act)]} \n Predicted:{candles[int(pred)]}',fontsize=10)
fig.tight_layout() 
plt.savefig('results.png') 
