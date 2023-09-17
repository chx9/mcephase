import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm
from PIL import Image
from scipy.signal import find_peaks
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms


from utils import read_config_file, Accumulator, read_mov
from utils.data import McePhaseDataset
from model.mcephase import AttentionMcePhase

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
# matplotlib.use('TkAgg')
tol = 1e-8
config_path = 'config.json'
config = read_config_file(config_path)
seq_len = config['data']['seq_len']

device = 'cuda'
data_type = 'test'

if not os.path.exists(f'results/figures/{data_type}'):
    os.makedirs(f'results/figures/{data_type}')
info_path = f'data/{data_type}_info.json'

info = read_config_file(f'data/{data_type}_info.json')
root_dir = 'frames'
patients = list(info.keys())
results = defaultdict(dict)
for idx in range(len(patients)):
    patient = patients[idx]
    patient_info = info[patients[idx]]
    indexs = patient_info['index']
    labels = patient_info['label']
    mce_len = indexs[1] - indexs[0]+1
    pred = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pred, std)
    ])
    accumulator = Accumulator(mce_len)
    imgs = []
    for i in range(indexs[0], indexs[1]+1):
        im_path = osp.join(root_dir, patient, f'{i}.png')
        im = Image.open(im_path)
        im = torchvision.transforms.Resize(128)(im)
        im = to_tensor(im)
        imgs.append(im)
    model = AttentionMcePhase().to(device)
    model.load_state_dict(torch.load(
        'checkpoints/epoch_200.pth')['model_state_dict'])
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, mce_len+1-seq_len), desc='Processing', ncols=80):
            start = i
            end = i+seq_len
            im_seq = imgs[start:end]
            input_seq = torch.stack(im_seq).unsqueeze(0).to(device)
            pred = model(input_seq)
            pred = pred.squeeze(dim=0)
            accumulator.add(pred.cpu().numpy(), i)
    pred = accumulator.mean()
    inverted_pred = 1 - pred

    pred_es = find_peaks(inverted_pred, prominence=0.03, distance=15, height=0.5)[0]
    pred_ed = find_peaks(pred, prominence=0.05, distance=18, height=0.5)[0]

    true_es = find_peaks(1-np.array(labels), prominence=0.03,
                     distance=15, height=0.5)[0]
    true_ed = find_peaks(np.array(labels), prominence=0.05,
                     distance=18, height=0.5)[0]

    results[patient]['predicted_es'] = pred_es.tolist()
    # plt.show()
    fig, ax = plt.subplots()

    ax.plot(pred, color='green')
    ax.plot(labels)
    ax.scatter(pred_es, [0]*len(pred_es), marker='x', color='blue')
    ax.scatter(true_es, [0]*len(true_es), marker='o', color='red')

    fig.savefig(f"results/figures/{data_type}/{patient}.png")
    plt.close(fig)
with open(f"results/{data_type}.json", "w") as file:
    json.dump(results, file)
