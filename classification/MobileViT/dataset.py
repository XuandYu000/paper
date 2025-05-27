import os

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# 数据增强
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(256), # modify the size of the image to fit the model. 224x224 here.
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256), # modify the size of the image to fit the model. 224x224 here.
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
    ])
}

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path, mode="train"):
        """
        :param images_path: 图片路径
        :param mode: 训练模式
        """
        super().__init__()
        self.path = images_path
        self.files = sorted([x for x in os.listdir(images_path) if x.endswith(".jpg")])
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 获取图片路径
        fname = os.path.join(self.path, self.files[idx])
        # 打开图片
        im = Image.open(fname)

        # 进行数据增强
        if self.mode == "train":
            im = data_transform[self.mode](im)
        else:
            im = data_transform[self.mode](im)

        # 获取标签
        try:
            label = int(self.files[idx].split("_")[0])
        except:
            label = -1 # test has no label

        return im, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        imgs, labels = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)
        return imgs, labels