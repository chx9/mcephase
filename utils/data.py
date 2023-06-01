import torch
import numpy as np
import time
import torchvision
import random
import os
import os.path as osp
import json
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from torchvision import transforms
import cv2
root_dir = 'frames'
seq_len = 30
info_path = '/home/u/Desktop/mcephase/train_info.json'


class McePhaseDataset(Dataset):
    def __init__(self, root_dir=root_dir, seq_len=seq_len, info_path=info_path):
        with open(info_path) as f:
            self.info = json.load(f)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.patients = list(self.info.keys())
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients[index]
        index = self.info[patient]['index']
        label = self.info[patient]['label']
        selected_index = random.randint(index[0], index[1])
        start_index = selected_index - seq_len // 2 + 1
        end_index = selected_index + seq_len // 2
        start_padding_num = max(0, index[0]-start_index)
        end_padding_num = max(0, end_index-index[1])
        imgs = []
        seed = int(time.time()) % 10000
        transform = A.Compose(
            [
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2)),
            A.Rotate(limit=2, border_mode=cv2.BORDER_CONSTANT, value=0),
            ]
        )
        resize = torchvision.transforms.Resize(256)
        for i in range(max(start_index, index[0]), min(end_index+1, index[1]+1)):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            img_path = osp.join(root_dir, patient, f'{i}.png')
            img = Image.open(img_path)
            img = resize(img)
            img = transform(image=np.array(img))['image']
            img = self.to_tensor(img)
            imgs.append(img)
        label = label[max(start_index-index[0], 0) : min(end_index-index[0]+1, len(label))]

        if start_padding_num > 0:
            imgs = start_padding_num*[torch.zeros((3, 256, 256))] + imgs
            label = start_padding_num*[0.0] + label
        if end_padding_num > 0:
            imgs = imgs+end_padding_num*[torch.zeros((3, 256, 256))]
            label = label + end_padding_num*[0.0]
        imgs = torch.stack(imgs, dim=0)
        label = torch.tensor(label).to(torch.float)
        assert imgs.shape[0] == self.seq_len and label.shape[0] == self.seq_len, f"{patient}, {selected_index}"
        return imgs, label


if __name__ == '__main__':
    info_path = '/home/u/Desktop/mcephase/train_info.json'
    data = McePhaseDataset()
    img, label = next(iter(data))
    exit()
