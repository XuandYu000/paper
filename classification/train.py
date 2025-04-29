# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tensorboard.summary.v1 import image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
# K-fold cross validation and boosting
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier

from torchvision.models import resnet18, resnet50, vgg16

from data.DataSet import FoodDataset
from models import VGG
import yaml
import argparse

def model_params(model):
    """
    Count the number of parameters in the model.
    Args:
        model (torch.nn.Module): The model to count parameters for.
    Returns:
        None
    """
    for param in model.parameters():
        print(param.size())

    # 统计模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def setLoss(loss_name):
    """
    Set the loss function based on the provided name.
    Args:
        loss_name (str): The name of the loss function.
    Returns:
        torch.nn.Module: The loss function.
    """
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model configuration")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")
    # Parse the model settings
    model_cfg = config["model"]
    model = VGG._vgg(cfg=model_cfg["name"],
                     batch_norm=model_cfg["batch_norm"],
                     num_classes=model_cfg["num_classes"],
                     dropout=model_cfg["dropout"],
                     init_weights=model_cfg["init_weights"],
                     ).to(device)

    train_cfg = config["train"]
    dataset_cfg = config["dataset"]
    loss_fn = setLoss(train_cfg["criterion"]["name"])
    if train_cfg["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["optimizer"]["weight_decay"])
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["optimizer"]["weight_decay"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    save_path = os.path.join(train_cfg["Resume"]["resume_path"], config["name"]+".pth")
    writer = SummaryWriter()

    # Create the dataset
    train_dataset = FoodDataset(path=dataset_cfg["train"]["path"], tfm="train")
    valid_dataset = FoodDataset(path=dataset_cfg["valid"]["path"], tfm=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    train_step = len(train_loader)
    best_acc = 0.0
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(valid_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_outputs = model(val_images.to(device))
                _, predicted = torch.max(val_outputs, 1)
                acc += (predicted == val_labels.to(device)).sum().item()

        val_acc = acc / len(valid_dataset)
        print(f"epoch {epoch + 1} validation accuracy: {val_acc:.3f}")
        writer.add_scalar("train_loss", running_loss / train_step, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)

        # Save the model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

