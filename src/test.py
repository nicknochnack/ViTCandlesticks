from model import ViT
from data import ClfDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from matplotlib import pyplot as plt 

candles = {
    0:"doji",
    1:"bullish_engulfing",
    2:"bearish_engulfing",
    3:"morning_star",
    4:"evening_star"
}

model = ViT()
model.load_state_dict(torch.load("checkpoints/50_model.pt", weights_only=True, map_location=torch.device('cpu')))
model.eval()

data = ClfDataset("data/test_data")
dataloader = DataLoader(
    data, batch_size=4, shuffle=True
    # , prefetch_factor=2, num_workers=2
)
sample = next(iter(dataloader))
# import sys; sys.exit()
X = sample[0] 
print(X.shape)
softy = torch.nn.Softmax(dim=1)
preds = model(X)

print(softy(preds)) 
argmaxes = torch.argmax(softy(preds), dim=-1)
print(argmaxes)
print(sample[1])

loss_fn = nn.CrossEntropyLoss()
print(loss_fn(preds, sample[1]))

fig, ax = plt.subplots(2,2) 
axs = ax.flatten()
for act, pred, img, ax in zip(sample[1], argmaxes, X, axs):
    img_np = img.permute(1, 2, 0).numpy()
    img_min, img_max = img_np.min(), img_np.max()
    img_np_scaled = (img_np - img_min) / (img_max - img_min)
    ax.imshow(img_np_scaled)
    ax.set_title(f'Actual:{candles[int(act)]} \n Predicted:{candles[int(pred)]}',fontsize=10)
fig.tight_layout() 
plt.savefig('results.png') 
