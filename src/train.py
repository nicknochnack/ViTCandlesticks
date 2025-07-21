# Bring in model for training 
from model import ViT
# Bring in dataset class
from data import ClfDataset
# Bring in torch for setting no grad and manual seed  
import torch
# Bring in nn for losses, optim for Adam and save for saving
from torch import nn, optim, save
# Bring in Dataloader for batching, random_split for train val, and default collate for mixup/cutmix pipeline
from torch.utils.data import DataLoader, random_split, default_collate
# This is my favourite library
from colorama import Fore
# Mixup/CutMix 
from torchvision.transforms import v2
# Outputting model info 
from torchinfo import summary

# Run the pipeline brev
if __name__ == "__main__":
    # Set seed for reproducability - I think that's a real word. 
    torch.manual_seed(42)
    # Setup train and test datasets
    train_val_data = ClfDataset("data/train_data", train=True)
    test_data = ClfDataset("data/test_data", train=False)
    # Split train_val into train and val subsets 
    train_size = int(0.7 * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size])

    # Implement cut_mix, mix_up for train only 
    cutmix = v2.CutMix(num_classes=5)
    mixup = v2.MixUp(num_classes=5)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup, v2.Identity()], p=[0.25,0.25,0.5])
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    train_dataset = DataLoader(
        train_data, batch_size=16, shuffle=True, collate_fn=collate_fn,
    )
    # Create val pipeline - used for hyper tuning and model val
    val_dataset = DataLoader(
        val_data, batch_size=16, shuffle=False, 
    )
    # Create test pipeline - used for assessing how well this big dawg is going to perform in the real world
    test_dataset = DataLoader(
        test_data, batch_size=16, shuffle=False,
    ) 
    # Instantiate the model 
    model = ViT()
    # Output model info 
    summary(model, (1, 3, 104, 72))

    # How long we'll train for, we'll capture a bunch of checkpoints so we can choose what to use 
    epochs = 100
    # Good ol cross entropy loss - note this applies the softmax internally 
    loss_fn = nn.CrossEntropyLoss()
    # Setup optimizer with a learning rate of 0.0001
    opt = optim.Adam(model.parameters(), lr=1e-4)
    # Add in a warm restart scheduler to slow learning rate the longer training goes on
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataset)*30, T_mult=2)

    # Used to show progress over entire dataset
    train_batches = len(train_dataset)
    print(Fore.LIGHTYELLOW_EX + "Starting training" + Fore.RESET)
    # Run training loop 
    for epoch in range(epochs):
        # Enable all layers 
        model.train()
        # Used to accumulate loss over the whole epoch
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_dataset):
            # Grab batch
            X, y = batch
            # Make preds
            yhat = model(X)
            # Calc loss
            loss = loss_fn(yhat, y)
            # Accumulate
            epoch_loss += loss.item()

            # Zero grads
            opt.zero_grad()
            # Calculate gradients
            loss.backward()
            # Apply them via backprop 
            opt.step()
            # Progress on the scheduler
            scheduler.step()

            # Fancy progress bar
            progress = (batch_idx + 1) / train_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{train_batches} batches",
                end="",
            )

        print(f" - Train Loss: {epoch_loss/train_batches:.4f}", end="")
        # Do the same stuff but for validation partition and test partition
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(val_dataset):
                X, y = batch
                yhat = model(X)
                loss = loss_fn(yhat, y)
                epoch_loss += loss.item()

            print(f" - Val Loss: {epoch_loss/len(val_dataset):.4f}", end="")

            for batch_idx, batch in enumerate(test_dataset):
                X, y = batch
                yhat = model(X)
                loss = loss_fn(yhat, y)
                epoch_loss += loss.item()

            print(f" - Test Loss: {epoch_loss/len(test_dataset):.4f}")
    
    # Save the model every 5 epochs
        if epoch % 5 == 0:
            save(model.state_dict(), f"checkpoints/{epoch}_model.pt")
    # And one final save for good measure 
    save(model.state_dict(), f"checkpoints/{epoch}_model.pt")