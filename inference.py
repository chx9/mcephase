import torch
import json
import numpy as np
import os.path as osp
import numpy as np
import os
from tqdm import tqdm
import torchvision
from torch import nn
from utils.data import McePhaseDataset
from torch.utils.data import DataLoader
from model.mcephase import AttentionMcePhase 
from matplotlib import pyplot as plt
from utils import read_config_file
from torchvision import transforms
from PIL import Image
from scipy.signal import find_peaks
from utils import Accumulator, read_mov, read_config_file
import matplotlib
from utils import read_config_file
matplotlib.use('TkAgg')
from collections import defaultdict
tol = 1e-8
config_path = '/home/u/Desktop/mcephase/config.json'
config = read_config_file(config_path) 
seq_len = config['data']['seq_len']

device = 'cuda'
data_type = 'train'

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
    model.load_state_dict(torch.load('checkpoints/epoch_100.pth')['model_state_dict'])
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
    inverted_pred =  1- pred 


    true_ed = np.where(np.isclose(labels, 1.0, rtol=tol, atol=tol))[0]
    true_es = np.where(np.isclose(labels, 0.0, rtol=tol, atol=tol))[0]

    ES = find_peaks(inverted_pred,prominence=0.03,distance=15, height=0.5)[0] 
    ED = find_peaks(pred,prominence=0.05,distance=18, height=0.5)[0]

    ES_ = find_peaks(1-np.array(labels),prominence=0.03,distance=15, height=0.5)[0] 
    ED_ = find_peaks(np.array(labels),prominence=0.05,distance=18, height=0.5)[0]
    
    results[patient]['true_es'] = true_es.tolist()
    results[patient]['predicted_es'] = ES.tolist()
    # plt.show()
    fig, ax = plt.subplots()

    ax.plot(pred, color='green')
    ax.plot(labels)
    ax.scatter(ES, [0]*len(ES), marker='x', color='blue')
    ax.scatter(true_es, [0]*len(true_es), color='red')
    ax.scatter(ES_, [0]*len(ES_), marker='o', color='red')

    fig.savefig(f"results/figures/{data_type}/{patient}.png")
    plt.close(fig) 
with open(f"results/{data_type}.json", "w") as file:
    json.dump(results, file)