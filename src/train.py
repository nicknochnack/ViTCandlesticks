from model import ViT
from data import ClfDataset
import torch
from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split, default_collate
from colorama import Fore
from torchvision.transforms import v2
from torchinfo import summary


if __name__ == "__main__":
    torch.manual_seed(42)
    # Get train and test data 
    train_val_data = ClfDataset("data/train_data", train=True)
    test_data = ClfDataset("data/test_data", train=False)
    # Create train and val splits
    train_size = int(0.7 * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size])

    # Implement cut_mix, mix_up for train and val only 
    cutmix = v2.CutMix(num_classes=5)
    mixup = v2.MixUp(num_classes=5)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    train_dataset = DataLoader(
        train_data, batch_size=16, shuffle=True, prefetch_factor=2, num_workers=2, collate_fn=collate_fn,
    )
    val_dataset = DataLoader(
        val_data, batch_size=16, shuffle=False, prefetch_factor=2, num_workers=2
    )
    test_dataset = DataLoader(
        test_data, batch_size=16, shuffle=False, prefetch_factor=2, num_workers=2
    ) 

    model = ViT().to('cuda')
    print(summary(model, (1, 3, 120, 72)))
    loss_fn = nn.CrossEntropyLoss()

    epochs = 400
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataset)*30, T_mult=2)
    train_batches = len(train_dataset)
    print(Fore.LIGHTYELLOW_EX + "Starting training" + Fore.RESET)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_dataset):
            X, y = batch
            yhat = model(X.to('cuda'))
            loss = loss_fn(yhat, y.to('cuda'))
            epoch_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            # Progress bar
            progress = (batch_idx + 1) / train_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{train_batches} batches",
                end="",
            )

        print(f" - Train Loss: {epoch_loss/train_batches:.4f}", end="")
        # Evaluate val and test set
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(val_dataset):
                X, y = batch
                yhat = model(X.to('cuda'))
                loss = loss_fn(yhat, y.to('cuda'))
                epoch_loss += loss.item()

            print(f" - Val Loss: {epoch_loss/len(val_dataset):.4f}", end="")

            for batch_idx, batch in enumerate(test_dataset):
                X, y = batch
                yhat = model(X.to('cuda'))
                loss = loss_fn(yhat, y.to('cuda'))
                epoch_loss += loss.item()

            print(f" - Test Loss: {epoch_loss/len(test_dataset):.4f}")

        if epoch % 25 == 0:
            save(model.state_dict(), f"checkpoints/{epoch}_model.pt")
    save(model.state_dict(), f"checkpoints/{epoch}_model.pt")