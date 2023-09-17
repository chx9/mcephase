import os
import os.path as osp
from io import BytesIO
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
from torchvision import transforms

from model.loss import CombineLoss
from model.mcephase import AttentionMcePhase

from utils import read_config_file, Accumulator
from utils.data import McePhaseDataset

from tqdm import tqdm
import argparse
from scipy.signal import find_peaks
from PIL import Image

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
seq_len = config['data']['seq_len']

# dataloader
train_data = McePhaseDataset(info_path=train_info_path)
test_data = McePhaseDataset(info_path=test_info_path, is_train=False)
train_dataloader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, drop_last=True,)
# writer = SummaryWriter()
now = datetime.now()
now_str = now.strftime("%Y%m%d-%H%M%S")
dir_name = os.path.join("runs/train_test", now_str)
os.makedirs(dir_name, exist_ok=True)
writer_train = SummaryWriter(os.path.join(dir_name, "train"))
writer_test = SummaryWriter(os.path.join(dir_name, "test"))
# writer = SummaryWriter()
model = AttentionMcePhase(n_features=1024).to(device)
# loss function
# criterion = nn.MSELoss()
criterion = CombineLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
resume_epoch = 0
if resume:

    checkpoint_files = os.listdir('checkpoints')
    checkpoint_file = sorted(
        checkpoint_files, key=lambda x: int(x.split('_')[-1][:-4]))[-1]
    checkpoint = torch.load(osp.join('checkpoints', checkpoint_file))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch = checkpoint['epoch']
def plot_fig(preds, labels, pred_es, true_es):

    fig, ax = plt.subplots()
    ax.plot(preds, color='green')
    ax.plot(labels)
    ax.scatter(pred_es, [0]*len(pred_es), marker='x', color='blue')
    ax.scatter(true_es, [0]*len(true_es), marker='o', color='red')
    if __debug__:
        plt.show(block=True)
    else:
        plt.show()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return np.array(Image.open(buf))
def plot_ndarray(data):
    """
    Plots a 1D NumPy ndarray as a line chart.

    Parameters:
    data (numpy.ndarray): The input 1D NumPy ndarray to be plotted.
    """

    # Check if the input is a NumPy ndarray
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy ndarray")

    # Check if the input data is 1D
    if data.ndim != 1:
        raise ValueError("Input data must be 1D")

    # Plot the data as a line chart
    plt.plot(data)

    # In debug mode, use plt.show(block=True) to keep the plot window open
    if __debug__:
        plt.show(block=True)
    else:
        # In non-debug mode, the default behavior of plt.show() will be used
        plt.show()

def eval_es_detection(model, writer, epoch):
    model.eval()
    tol = 1e-8
    data_type = 'test'
    info = read_config_file(f'data/{data_type}_info.json')

    root_dir = 'frames'
    patients = list(info.keys())

    for idx in range(len(patients)//10) :
        patient = patients[idx]
        patient_info = info[patients[idx]]
        indexs = patient_info['index']
        labels = patient_info['label']
        mce_len = indexs[1] - indexs[0]+1
        preds = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(preds, std)
        ])
        accumulator = Accumulator(mce_len)
        imgs = []
        for i in range(indexs[0], indexs[1]+1):
            im_path = osp.join(root_dir, patient, f'{i}.png')
            im = Image.open(im_path)
            im = torchvision.transforms.Resize(128)(im)
            im = to_tensor(im)
            imgs.append(im)
        with torch.no_grad():
            for i in tqdm(range(0, mce_len+1-seq_len), desc='Processing', ncols=80):
                start = i
                end = i+seq_len
                im_seq = imgs[start:end]
                input_seq = torch.stack(im_seq).unsqueeze(0).to(device)
                preds = model(input_seq)
                preds = preds.squeeze(dim=0)
                accumulator.add(preds.cpu().numpy(), i)
        preds = accumulator.mean()
        inverted_pred = 1 - preds

        pred_es = find_peaks(inverted_pred, prominence=0.03, distance=15, height=0.5)[0]
        pred_ed = find_peaks(preds, prominence=0.05, distance=18, height=0.5)[0]


        true_es = find_peaks(1-np.array(labels), prominence=0.03,
                        distance=15, height=0.5)[0]
        true_ed = find_peaks(np.array(labels), prominence=0.05,
                        distance=18, height=0.5)[0]
        # plot_ndarray(preds)
        plot = plot_fig(preds, labels, pred_es, true_es)
        writer.add_image(f'patient_{idx}', plot, dataformats='HWC', global_step=epoch)
def forward_step(model, images, labels, criterion, mode=''):
    if mode == 'test':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)

    loss = criterion(output, labels)

    if mode == 'test':
        # Calculate the differences within tolerance for each batch
        valid_differences = []
        tolerance = 10

        for i in range(output.shape[0]):
            model_output = output[i].cpu().numpy()
            true_label_batch = labels[i].cpu().numpy()

            for true_label in true_label_batch:
                diffs = np.abs(model_output - true_label)

                # If the minimum difference is within the tolerance
                if np.min(diffs) <= tolerance:
                    valid_differences.append(np.min(diffs))

        # Calculate the average difference and standard deviation
        if len(valid_differences) > 0:
            average_difference = np.mean(valid_differences)
            std_difference = np.std(valid_differences)
        else:
            average_difference = np.nan
            std_difference = np.nan

        return loss, output.detach().cpu(), average_difference, std_difference

    else:
        return loss, output.detach().cpu()


for epoch in range(resume_epoch, resume_epoch+epochs):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, pred = forward_step(model, x, y, criterion, mode='train')
        loss.backward()
        optimizer.step()
        # writer.add_scalar('train_step_loss', loss.item(), epoch*(len(train_dataloader)+i))
        epoch_loss += loss.item()

    print('='*20)
    writer_train.add_scalar('loss', epoch_loss/len(train_dataloader), epoch+1)
    print('train epoch: {}, loss:{}'.format(
        epoch+1, epoch_loss/len(train_dataloader)))
    print('='*20)
    # validation

    model.eval()
    if (epoch+1) % test_interval == 0:
        epoch_loss = 0
        epoch_avg_diffs = []
        epoch_std_diffs = []

        for i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            loss, pred, average_difference, std_difference = forward_step(
                model, x, y, criterion, mode='test')

            epoch_loss += loss.item()

            if average_difference is not None:
                epoch_avg_diffs.append(average_difference)

            if std_difference is not None:
                epoch_std_diffs.append(std_difference)

        epoch_loss /= len(test_dataloader)

        if epoch_avg_diffs:
            avg_epoch_avg_diff = np.mean(epoch_avg_diffs)
            avg_epoch_std_diff = np.mean(epoch_std_diffs)
        else:
            avg_epoch_avg_diff = np.nan
            avg_epoch_std_diff = np.nan

        print('*'*20)
        writer_test.add_scalar('loss', epoch_loss, epoch+1)
        writer_test.add_scalar('avg_diff', avg_epoch_avg_diff, epoch+1)
        writer_test.add_scalar('std_diff', avg_epoch_std_diff, epoch+1)
        print('test epoch: {}, loss: {}, avg diff: {}, std diff: {}'.format(
            epoch+1, epoch_loss, avg_epoch_avg_diff, avg_epoch_std_diff))
        print('*'*20)

        eval_es_detection(model, writer_test, epoch+1)

    if (epoch+1) % checkpoint_interval == 0:
        # save model
        if not osp.exists(osp.join('checkpoints')):
            os.mkdir(osp.join('checkpoints'))
        checkpoint_path = osp.join(
            'checkpoints', 'epoch_{}.pth'.format(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

writer_train.close()
writer_test.close()
