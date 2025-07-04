from model import ViT
from data import ClfDataset
from torch.utils.data import DataLoader
import torch
from torch import nn

model = ViT().to("cuda")
model.load_state_dict(torch.load("checkpoints/25_model.pt", weights_only=True))
model.eval()

data = ClfDataset("consolidated_data")
dataloader = DataLoader(
    data, batch_size=64, shuffle=True, prefetch_factor=2, num_workers=2
)
sample = next(iter(dataloader))
X = sample[0]  # torch.unsqueeze(sample[0], dim=0)
print(X.shape)
softy = torch.nn.Softmax(dim=1)
preds = model(X.to("cuda"))

print(torch.argmax(softy(preds), dim=-1))
print(sample[1])

loss_fn = nn.CrossEntropyLoss()

print(loss_fn(preds, sample[1].to("cuda")))
