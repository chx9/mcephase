import torch
import os.path as osp
import numpy as np
import os
from torch import nn
from utils.data import McePhaseDataset
from utils import read_mov
from torch.utils.data import DataLoader
from model.mcephase import McePhase
from matplotlib import pyplot as plt
from utils import read_config_file
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.signal import find_peaks
tol = 1e-8
class Accumulator:
    def __init__(self, length):
        self.length = length
        self.data = np.zeros((length,))
        self.counts = np.zeros((length,))
    
    def add(self, values, index):
        assert values.shape[0] <= self.length, f"Expected values of shape ({self.length},) or less, but got shape {values.shape}"
        length = min(self.length, values.shape[0])
        self.data[index:index+length] += values[:length]
        self.counts[index:index+length] += 1
    
    def mean(self):
        return self.data/self.counts
seq_len = 30
device = 'cuda'
test_info = read_config_file('test_info.json')
root_dir = 'frames'
patients = list(test_info.keys())
idx = 5
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

model = McePhase().to(device)
model.load_state_dict(torch.load('checkpoints/epoch_600.pth')['model_state_dict'])
model.eval()
mov_file_path = 'patient.mov'
imgs = read_mov(mov_file_path)
mec_len = len(imgs)
with torch.no_grad():
    for i in range(0, mce_len+1-seq_len):
        print(i)
        start = i
        end = i+seq_len
        im_seq = imgs[start:end]
        input_seq = torch.stack(im_seq).unsqueeze(0).to(device)
        pred = model(input_seq)
        accumulator.add(pred.cpu().numpy(), i)
        # break
pred = accumulator.mean()
inverted_pred =  pred * -1
ES = find_peaks(inverted_pred,prominence=0.03,distance=15)[0]
ED = find_peaks(pred,prominence=0.05,distance=18)[0]
fig, ax = plt.subplots()
ax.plot(pred)


# true_ed = np.where(np.isclose(labels, 1.0, rtol=tol, atol=tol))[0]
# true_es = np.where(np.isclose(labels, 0.0, rtol=tol, atol=tol))[0]
print("zeros")
print("ones")
plt.show()
fig.savefig("plot.png")
exit()