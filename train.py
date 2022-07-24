from multiprocessing import freeze_support
from net import  Network
from dataset import GoData
from torch.utils.data import DataLoader
import torch
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.BCELoss()

 
    def forward(self, x, y):
        b, w = x
        bt, wt = y
        return self.loss1(b,bt) + self.loss2(w,wt)


if __name__=="__main__":
    freeze_support()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model
    model  = Network().to(device)

    # dataloader
    train_loader = DataLoader(dataset=GoData('train'), batch_size=32)
    val_loader = DataLoader(dataset=GoData('val'), batch_size=32)



    # metric
    criterion = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    n_total_steps = len(train_loader)
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            x, c = images
            x = x.to(device)
            c = c.to(device)
            bt,wt = labels#.to(device)
            bt = bt.to(device)
            wt = wt.to(device)
            # init optimizer
            optimizer.zero_grad()
            
            # forward -> backward -> update
            outputs = model(x,c)
            loss = criterion(outputs, (bt,wt))
            loss.backward()

            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'epoch {epoch+1}/{10}, step {i+1}/{n_total_steps}, loss = {loss.item():.10f}')
            if (i + 1) % 250 == 0:
                torch.save(model.state_dict(), 'model.weight')

    print('Finished Training')