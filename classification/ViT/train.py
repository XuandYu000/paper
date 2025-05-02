import math
import os
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import vit_base_patch16, vit_large_patch16, vit_huge_patch14

# 数据增强
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224), # modify the size of the image to fit the model. 224x224 here.
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.55807906, 0.45261728, 0.34557677], std=[0.23075283, 0.24137004, 0.24039967])
    ]),
    "val": transforms.Compose([
        transforms.Resize(224), # modify the size of the image to fit the model. 224x224 here.
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

def get_cosine_schedule_with_warmup(
	optimizer: torch.optim.Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

def parse_args():
    """:arguments"""
    config = {
        # model config
        "gpu_id": 0,
        "model_name": "vit_base_patch16", # vit_base_patch16, vit_large_patch16, vit_huge_patch14
        "num_classes": 11,

        # training config
        "num_epochs": 80,
        "warmup_epochs": 8,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "loss_function": nn.CrossEntropyLoss(),
        "optimizer": optim.AdamW,
    }
    return config

def main(
        gpu_id,
        model_name,
        num_classes,

        num_epochs,
        warmup_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        loss_function,
        optimizer,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    writer = SummaryWriter()
    save_pth = "./{}.pth".format(model_name)

    # 加载数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../datasets"))
    image_path = os.path.join(data_root, "food11")
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
    if model_name == "vit_base_patch16":
        model = vit_base_patch16(num_classes=num_classes)
    elif model_name == "vit_large_patch16":
        model = vit_large_patch16(num_classes=num_classes)
    elif model_name == "vit_huge_patch14":
        model = vit_huge_patch14(num_classes=num_classes)
    else:
        raise ValueError("model name must be mobilenetv3_small or mobilenetv3_large. Got {}".format(model_name))

    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # schedule the learning rate
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_epochs * len(train_loader),
        num_training_steps=num_epochs * len(train_loader),
    )

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

            outputs = model(imgs.to(device))
            loss = loss_function(outputs, labels.to(device))

            # update model
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{num_epochs}] loss:{loss:.3f} "
            writer.add_scalar("train_loss", loss, epoch * train_steps + step)
            writer.add_scalar("learning_rate", optimizer.state_dict()['param_groups'][0]['lr'], epoch * train_steps + step)

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
    main(**parse_args())