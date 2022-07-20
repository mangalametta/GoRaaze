from multiprocessing import freeze_support
from net import Network
from dataset import GoData
from torch.utils.data import DataLoader
import torch
from torch import nn


if __name__=="__main__":
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model
    model  = Network().to(device)

    # dataloader
    train_loader = DataLoader(dataset=GoData('train'), batch_size=64)
    val_loader = DataLoader(dataset=GoData('val'), batch_size=64)



    # metric
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    n_total_steps = len(train_loader)
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # init optimizer
            optimizer.zero_grad()
            
            # forward -> backward -> update
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'epoch {epoch+1}/{10}, step {i+1}/{n_total_steps}, loss = {loss.item():.10f}')
                torch.save(model.state_dict(), 'model.weight')

    print('Finished Training')