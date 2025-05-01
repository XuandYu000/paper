import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import resnet34, resnet50, resnet101
from torchvision.models import resnet34 as Resnet34


# 数据增强
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
    ])
}

class FoodDataSet(Dataset):
    def __init__(self, path, tfm=None, files=None):
        super().__init__()
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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # config
    # model config
    model_name = "resnet34"
    num_classes = 11

    # training config
    num_epochs = 300
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-5
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    writer = SummaryWriter()
    save_pth = "./{}.pth".format(model_name)

    # 加载数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    image_path = os.path.join(data_root, "datasets", "food11")
    assert os.path.exists(image_path), "dataset path does not exist."

    # 加载数据集
    train_dataset = FoodDataSet(os.path.join(image_path, "train"), data_transform["train"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)

    val_dataset = FoodDataSet(os.path.join(image_path, "validation"), data_transform["val"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # create model
    if model_name == "resnet34":
        model = resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        model = resnet101(num_classes=num_classes)

    model = Resnet34(num_classes=num_classes)
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # training
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            imgs, labels = data
            # 将梯度清零
            optimizer.zero_grad()
            outputs = model(imgs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{num_epochs}] loss:{loss:.3f} "
            writer.add_scalar("train_loss", loss, epoch * train_steps + step)
            writer.add_scalar("learning_rate", learning_rate, epoch * train_steps + step)

        # validate
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for step, val_data in enumerate(val_bar):
                val_imgs, val_labels = val_data
                outputs = model(val_imgs.to(device))
                pred_y = torch.max(outputs, dim=1)[1]
                acc += (pred_y == val_labels.to(device)).sum().item()

                val_bar.desc = f"valid epoch[{epoch + 1}/{num_epochs}]"
                writer.add_scalar("valid_acc", acc / val_num, epoch * train_steps + step)
        val_acc = acc / val_num
        print(f"epoch {epoch + 1} train loss: {running_loss / train_steps:.3f}, val acc: {val_acc:.3f}")

        # save model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_pth)
            print(f"Get best model saved to {save_pth}.")

    print("Training complete.")

if __name__ == '__main__':
    main()