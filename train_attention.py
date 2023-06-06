import torch
import os.path as osp
import os
from torch import nn
from utils.data import McePhaseDataset
from torch.utils.data import DataLoader
from model.mcephase import McePhase, AttentionMcePhase
from utils import read_config_file
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(description='args')
parser.add_argument('--resume', action='store_true',
                    help='resume training from a saved checkpoint')
args = parser.parse_args()
resume = args.resume
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = './config.json'
config = read_config_file(config_path) 
# train intervals
checkpoint_interval = config['train']['checkpoint_interval']
test_interval = config['test']['test_interval']
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
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,)
writer = SummaryWriter()
# model

model = AttentionMcePhase().to(device)
# loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
resume_epoch = 0
if resume:

    checkpoint_files = os.listdir('checkpoints_attention')
    checkpoint_file = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1][:-4]))[-1]
    checkpoint = torch.load(osp.join('checkpoints_attention', checkpoint_file))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch = checkpoint['epoch']

def forward_step(model, images, labels, critierion, mode=''):
    if mode == 'test':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)
    loss = criterion(output, labels)
    pred =output.detach()
    return loss, pred.cpu()
for epoch in range(resume_epoch, resume_epoch+epochs):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, pred = forward_step(model, x, y, criterion, mode='train')
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_step_loss', loss.item(), epoch*(len(train_dataloader)+i))
        epoch_loss += loss.item()

    print('='*20)
    writer.add_scalar('epoch_train_loss', epoch_loss/len(train_dataloader), epoch+1)
    print('train epoch: {}, loss:{}'.format(epoch+1, epoch_loss/len(train_dataloader)))
    print('='*20)
      # validation 

    model.eval() 
    if (epoch+1) % test_interval == 0:
        epoch_loss = 0
        for i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            loss, pred = forward_step(model, x, y, criterion, mode='test')
            epoch_loss += loss.item()
        # epoch loss
        print('*'*20)
        writer.add_scalar('epoch_test_loss', epoch_loss/len(test_dataloader), epoch+1)
        print('test epoch: {}, loss:{}'.format(epoch+1, epoch_loss/len(test_dataloader)))
        print('*'*20)
    
    if (epoch+1) % checkpoint_interval == 0:
        # save model
        if not osp.exists(osp.join('checkpoints_attention')):
             os.mkdir(osp.join('checkpoints_attention'))
        checkpoint_path =  osp.join('checkpoints_attention', 'epoch_{}.pth'.format(epoch+1))
        torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    }, checkpoint_path)
