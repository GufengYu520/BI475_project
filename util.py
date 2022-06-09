import os
import random
from torchvision.io import read_video
import numpy as np
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset
from torchvision import datasets


def getVids(data_path):
    vids = []
    labels = []
    for sub_dir in os.listdir(data_path):
        if sub_dir == 'light':
            label = 0
            dir_class = os.path.join(data_path, sub_dir)
            for file in os.listdir(dir_class):
                labels.append(label)
                vids.append(getOneVid(os.path.join(dir_class, file)))

        elif sub_dir == 'moderate':
            label = 1
            dir_class = os.path.join(data_path, sub_dir)
            for file in os.listdir(dir_class):
                labels.append(label)
                vids.append(getOneVid(os.path.join(dir_class, file)))

        elif sub_dir == 'vigorous':
            label = 2
            dir_class = os.path.join(data_path, sub_dir)
            for file in os.listdir(dir_class):
                labels.append(label)
                vids.append(getOneVid(os.path.join(dir_class, file)))

    return torch.stack(vids, dim=0).to(torch.float32), torch.tensor(labels)

# 归一化
mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]

def preprocess(vid):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=mean, std=std),
            T.Resize(size=(112, 112)),
        ]
    )
    vid = transforms(vid)
    return vid


def getOneVid(data_path):
    """随机选择16帧"""
    frames, _, _ = read_video(data_path)
    frames = frames.permute(0, 3, 1, 2)
    if len(frames) >= 16:
        indexs = list(range(len(frames)))
        random.shuffle(indexs)
        index_16 = sorted(indexs[:16])

        return preprocess(frames[index_16])
    elif len(frames) > 0:
        frames = frames.to(torch.float32)
        avg = torch.mean(frames, dim=0)
        avgs = avg.repeat(16-len(frames), 1, 1, 1)

        return preprocess(torch.cat((frames, avgs), 0))

    else:
        return preprocess(torch.randn(16, 3, 112, 112))


class vidData(Dataset):
    def __init__(self, data_path):
        super(vidData, self).__init__()
        # self.vids, self.labels = getVids(data_path)
        self.dataset = datasets.DatasetFolder(data_path, getOneVid, extensions=('.mp4'))


    def __getitem__(self, item):
        vid, label = self.dataset[item]
        return vid, label

    def __len__(self):
        return len(self.dataset)

