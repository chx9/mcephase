import torch
from torch import nn
from utils.data import McePhaseDataset
from torch.utils.data import DataLoader
from model.mcephase import McePhase
from utils import read_config_file
import torch.optim.lr_scheduler as lr_scheduler
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = './config.json'
config = read_config_file(config_path) 
# train intervals
checkpoint_interval = config['train']['checkpoint_interval']
log_interval = config['train']['log_interval']
# hyper params
batch_size = config['train']['batch_size']
lr = config['train']['lr']
num_workers = config['data']['num_workers']
train_info_path = config['train']['train_info_path']
test_info_path = config['test']['test_info_path']
epochs = config['train']['epochs']

# dataloader
train_data = McePhaseDataset(info_path=train_info_path)
test_data = McePhaseDataset(info_path=test_info_path, is_train=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,)

# model
model = McePhase().to(device)

# loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)




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
    y = y.squeeze(dim=0)
    model.train()
    trainer.zero_grad()
    pred = model(X)
    l = loss_fn(pred, y)
    l.backward()
    trainer.step()
    return l
for epoch in range(epochs):
    batch_loss_sum = torch.zeros(1).to(device)
    model.lstm.hidden_state = None
    for i, (features, labels) in enumerate(train_loader):
        loss = train_batch(model, features, labels, criterion, optimizer, device) 
        batch_loss_sum += loss
        if i % 10 == 0:
            print(f'epoch {epoch}, {i} train_loss {batch_loss_sum.item() / (i+1)}')

    if epoch % 5 == 0:
        test_loss = cal_testloss(model, test_loader, criterion)
        print(f'epoch { epoch } test loss: {test_loss.item()}')
    scheduler.step() 
torch.save(model.state_dict(), 'model.pth')
