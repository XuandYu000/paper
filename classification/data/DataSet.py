import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

train_tfm = transforms.Compose([
    # transforms.Resize((128, 128)),

    # By XuandYu000
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    # By XuandYu000
    transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
test_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((128, 128)),
    # You may add some transforms here.
    # By XuandYu000
    transforms.Resize(256),
    transforms.CenterCrop(224),

    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    # By XuandYu000
    transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
])

class FoodDataset(Dataset):

    def __init__(self, path, tfm=train_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label


class MyDataset(Dataset):

    def __init__(self, files, path, tfm=test_tfm):
        super(MyDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in files if x.endswith(".jpg")])
        self.trasform = tfm
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.trasform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1
        return im, label