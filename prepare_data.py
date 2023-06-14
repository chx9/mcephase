import os
import json
import cv2
import os.path as osp
import av
import json
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import defaultdict
def extract_frames_from_video(video_path):
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frames.append(frame)
    video_capture.release()
    return frames

tau = 3
delta = 1
amp = 0.2
def cal_systolic(ed, es):
    """
        [ed, es)
    """
    x = np.linspace(ed, es-1, es-ed)
    y = np.power( np.fabs(x-(es)) / np.fabs(es-ed), tau)
    y[0] = 1+amp
    return y
def cal_diastolic(es, ed):
    """
        [es, ed)
    """
    x = np.linspace(es, ed-1, ed-es)
    y = delta* np.power( np.fabs(x-(es)) / np.fabs(es-(ed)), 1/tau)
    y[0] = -amp
    return y
root_dir = 'videos'
videos = os.listdir(root_dir)
frame_info = defaultdict(dict)
for video in videos:
    mov_path = osp.join(root_dir, video)
    metadata = av.open(mov_path).metadata['com.apple.quicktime.description']
    metadata = json.loads(metadata)['v1.0']['selected']['0']
    frames = extract_frames_from_video(mov_path)
    frame_ids = [int(i) for i in metadata.split(',')]
    frame_index = list(range(frame_ids[0], frame_ids[-1]+1))
    # for frame_id in frame_index:
    #     folder_name = osp.join('frames', osp.basename(mov_path))
    #     if not osp.exists(folder_name):
    #         os.makedirs(folder_name)
    #     frame_filename = os.path.join(folder_name, f"{frame_id}.png")
    #     cv2.imwrite(frame_filename, frames[frame_id-1])
    y = np.array([])
    is_systolic = False
    for i in range(1, len(frame_ids)):
        if is_systolic:
            y_ = cal_systolic(frame_ids[i-1], frame_ids[i])
            is_systolic = False
            y = np.concatenate([y, y_])
        else:
            y_ = cal_diastolic(frame_ids[i-1], frame_ids[i])
            is_systolic = True
            y = np.concatenate([y, y_])
    if is_systolic:
        y = np.append(y, 1 + amp)
    else:
        y = np.append(y, -amp)
    frame_info[video]['index'] = [frame_ids[0], frame_ids[-1]]
    frame_info[video]['label'] = y.tolist()
frame_info = dict(frame_info)
with open('frame_info.json', 'w') as f:
    json.dump(frame_info, f)
exit()