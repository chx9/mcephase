import json
import numpy as np
from PIL import Image

import cv2
import torchvision
from torchvision import transforms


def read_config_file(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    return json_data


def read_mov(mov_file_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cap = cv2.VideoCapture(mov_file_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image = torchvision.transforms.Resize(128)(pil_image)
        im = to_tensor(pil_image)
        frames.append(im)
    cap.release()
    return frames


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
