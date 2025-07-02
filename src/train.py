from model import ViT
from data import ClfDataset 
import torch
from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split
from colorama import Fore 
import torchvision


if __name__ == '__main__': 
    data = ClfDataset('consolidated_data') 
    train_size = int(0.7 * len(data)) 
    test_size = len(data) - train_size 
    train_data, test_data = random_split(data, [train_size, test_size])
    train_dataset = DataLoader(train_data, batch_size=64, shuffle=True, prefetch_factor=2, num_workers=2)
    test_dataset = DataLoader(test_data, batch_size=64, shuffle=True, prefetch_factor=2, num_workers=2)

    model = ViT().to('cuda')
    # Load pretrained weights
    pretrained = torchvision.models.vit_b_16() 
    print(model.encoder.layers.load_state_dict(pretrained.encoder.layers.state_dict()))
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.11).to('cuda'))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.025778403,0.085235044,0.176152424,0.170470087,0.264228636,0.278135406]).to('cuda'))

    epochs=400
    opt = optim.Adam(model.parameters(), lr=0.00001)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
    # warmup_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=0.033, total_iters=30)
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

            # if epoch < 30: 
                # warmup_scheduler.step() 
    
            # Progress bar 
            progress = (batch_idx +1) / train_batches
            bar_length = 30 
            filled_length = int(bar_length * progress) 
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{train_batches} batches', end='')

        # scheduler.step() 
        print(f' - Train Loss: {epoch_loss/train_batches:.4f}', end='')
        # Evaluate test set
        model.eval() 
        with torch.no_grad():
            epoch_loss = 0.0  
            for batch_idx, batch in enumerate(test_dataset): 
                X, y = batch
                yhat = model(X.to('cuda'))
                loss = loss_fn(yhat, y.to('cuda')) 
                epoch_loss += loss.item() 

            print(f' - Test Loss: {epoch_loss/len(test_dataset):.4f}')

        if epoch % 25 == 0: 
            save(model.state_dict(), f'checkpoints/{epoch}_model.pt')
    save(model.state_dict(), f'checkpoints/{epoch}_model.pt')

