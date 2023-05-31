import torch
from torch import nn
from utils.data import McePhaseDataset
from torch.utils.data import DataLoader
from model.mcephase import McePhase
import torch.optim.lr_scheduler as lr_scheduler
batch_size = 4
num_workers = 2
device = 'cuda'
train_info_path = 'train_info.json'
test_info_path = 'test_info.json'
train_data = McePhaseDataset(info_path=train_info_path)
test_data = McePhaseDataset(info_path=test_info_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
model = McePhase().to(device)
criterion = nn.MSELoss()
num_epochs, lr, wd, device = 30, 1e-3, 1e-3, device
trainer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
scheduler = lr_scheduler.StepLR(trainer, step_size=50, gamma=0.1)
def cal_testloss(model, test_loader, loss_fn):
    loss_sum = torch.zeros(1).to(device)
    model.eval()
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            labels = labels.to(device)
            pred = model(features)
            l = loss_fn(pred, labels)
            loss_sum += l
    return loss_sum / len(test_loader)
def train_batch(model, X, y, loss_fn, trainer, device):
    X = X.to(device)
    y = y.to(device)
    model.train()
    trainer.zero_grad()
    pred = model(X)
    l = loss_fn(pred, y)
    l.backward()
    trainer.step()
    return l
for epoch in range(num_epochs):
    batch_loss_sum = torch.zeros(1).to(device)
    for i, (features, labels) in enumerate(train_loader):
        loss = train_batch(model, features, labels, criterion, trainer, device) 
        batch_loss_sum += loss
        if i % 10 == 0:
            print(f'epoch {epoch},i {i}train_loss: {batch_loss_sum.item() / (i+1)}')

    if epoch % 5 == 0:
        test_loss = cal_testloss(model, test_loader, criterion)
        print(f'epoch { epoch } test loss: {test_loss.item()}')
    scheduler.step() 
torch.save(model.state_dict(), 'model.pth')
