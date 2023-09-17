import os
import os.path as osp
import time
import random
import json

import cv2
import numpy as np
from PIL import Image
import albumentations as A

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from . import read_config_file


config_path = 'config.json'
config = read_config_file(config_path)
seq_len = config['data']['seq_len']
root_dir = config['data']['root_dir']
crop_size = config['data']['crop_size']
info_path = 'data/frame_info.json'


class McePhaseDataset(Dataset):
    def __init__(self, root_dir=root_dir, seq_len=seq_len, info_path=info_path, is_train=True):
        with open(info_path) as f:
            self.info = json.load(f)
        self.root_dir = root_dir
        self.is_train = is_train
        self.seq_len = seq_len
        self.patients = list(self.info.keys())
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.augmentation = A.Compose([
            A.RandomResizedCrop(
                height=crop_size, width=crop_size, scale=(0.8, 1.2)),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0),
        ])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):
        patient = self.patients[index]
        index = self.info[patient]['index']
        label = self.info[patient]['label']
        begin = index[0] + seq_len//2 - 1
        end = index[1] - seq_len//2
        selected_index = random.randint(min(begin, end), max(begin, end))
        # selected_index = random.randint(index[0] , index[1]) # add padding
        start_index = selected_index - seq_len // 2 + 1
        end_index = selected_index + seq_len // 2
        start_padding_num = max(0, index[0]-start_index)
        end_padding_num = max(0, end_index-index[1])
        imgs = []
        seed = int(time.time()) % 10000
        resize = torchvision.transforms.Resize(crop_size)
        for i in range(max(start_index, index[0]), min(end_index+1, index[1]+1)):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            img_path = osp.join(root_dir, patient, f'{i}.png')
            img = Image.open(img_path)
            img = resize(img)
            # augmentation
            if self.is_train:
                img = self.augmentation(image=np.array(img))['image']
            img = self.to_tensor(img)
            imgs.append(img)
        label = label[max(start_index-index[0], 0)                      : min(end_index-index[0]+1, len(label))]
        # add padding
        if start_padding_num > 0:
            imgs = start_padding_num * \
                [torch.zeros((3, crop_size, crop_size))] + imgs
            label = start_padding_num*[0.0] + label
        if end_padding_num > 0:
            imgs = imgs+end_padding_num * \
                [torch.zeros((3, crop_size, crop_size))]
            label = label + end_padding_num*[0.0]

        imgs = torch.stack(imgs, dim=0)
        label = torch.tensor(label).to(torch.float)

        assert imgs.shape[0] == self.seq_len and label.shape[
            0] == self.seq_len, f"{patient}, {selected_index}"

        return imgs, label


if __name__ == '__main__':
    info_path = '/home/u/Desktop/mcephase/data/train_info.json'
    data = McePhaseDataset()
    img, label = next(iter(data))
    exit()
