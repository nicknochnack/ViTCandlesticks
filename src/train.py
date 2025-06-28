from model import ViT
from data import ClfDataset 
from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split
from colorama import Fore 

if __name__ == '__main__': 
    data = ClfDataset('consolidated_data') 
    train_size = int(0.9 * len(data)) 
    test_size = len(data) - train_size 
    train_data, test_data = random_split(data, [train_size, test_size])
    train_dataset = DataLoader(train_data, batch_size=32, shuffle=True, prefetch_factor=2, num_workers=2)
    test_dataset = DataLoader(test_data, batch_size=32, shuffle=True, prefetch_factor=2, num_workers=2)

    model = ViT()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    epochs=40
    train_batches = len(train_dataset)
    print(Fore.LIGHTYELLOW_EX + "Starting training" + Fore.RESET) 
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0.0  
        for batch_idx, batch in enumerate(train_dataset): 
            X, y = batch
            yhat = model(X)
            loss = loss_fn(yhat, y) 
            epoch_loss += loss.item() 

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Progress bar 
            progress = (batch_idx +1) / train_batches
            bar_length = 30 
            filled_length = int(bar_length * progress) 
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{train_batches} batches', end='')

        print(f' - Train Loss: {epoch_loss/train_batches:.4f}', end='')
        # Evaluate test set
        model.eval() 
        epoch_loss = 0.0  
        for batch_idx, batch in enumerate(test_dataset): 
            X, y = batch
            yhat = model(X)
            loss = loss_fn(yhat, y) 
            epoch_loss += loss.item() 

        print(f' - Test Loss: {epoch_loss/len(test_dataset):.4f}')

        if epoch % 10 == 0: 
            save(model.state_dict(), f'checkpoints/{epoch}_model.pt')
    save(model.state_dict(), f'checkpoints/{epoch}_model.pt')

