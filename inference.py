import torch
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
matplotlib.use('TkAgg')

tol = 1e-8
seq_len = 30
device = 'cuda'
test_info = read_config_file('data/train_info.json')
root_dir = 'frames'
patients = list(test_info.keys())
# idx = 8
# patient = patients[idx]
for idx in range(len(patients)):
    patient = patients[idx]
    patient_info = test_info[patients[idx]]
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
    model.load_state_dict(torch.load('checkpoints/epoch_140.pth')['model_state_dict'])
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
    inverted_pred =  pred * -1
    ES = find_peaks(inverted_pred,prominence=0.03,distance=15)[0]
    ED = find_peaks(pred,prominence=0.05,distance=18)[0]
    fig, ax = plt.subplots()
    ax.plot(pred)


    true_ed = np.where(np.isclose(labels, 1.0, rtol=tol, atol=tol))[0]
    true_es = np.where(np.isclose(labels, 0.0, rtol=tol, atol=tol))[0]
    print('='*20)
    print(true_es[1:-1])
    print(ES)
    print(true_ed)
    print(ED)
    print('='*20)
    # plt.show()
    fig.savefig(f"figures/{patient}.png")