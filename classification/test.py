import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from tqdm.auto import tqdm
from data.DataSet import FoodDataset, MyDataset
from models import VGG
import yaml
import argparse

if __name__ == "__main__":
    # 设置GPU为1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser(description="Model configuration")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")

    # General setting
    train_cfg = config["train"]
    batch_size = train_cfg["batch_size"]
    save_path = os.path.join(train_cfg["Resume"]["resume_path"], config["name"]+".pth")

    # Load the model
    model_cfg = config["model"]
    model_cfg["init_weights"] = False
    model = VGG._vgg(cfg=model_cfg["name"],
                     batch_norm=model_cfg["batch_norm"],
                     num_classes=model_cfg["num_classes"],
                     dropout=model_cfg["dropout"],
                     init_weights=model_cfg["init_weights"],
                     ).to(device)
    # Load the model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Model loaded from {save_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {save_path}")

    # Load the dataset
    test_dataset = MyDataset(path=config["dataset"]["test"]["path"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Set the model to evaluation mode
    preds = {}
    model.eval()
    test_step = len(test_loader)
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, _, ids = test_data
            test_outputs = model(test_images.to(device))
            test_outputs = torch.sigmoid(test_outputs)
            outputs = test_outputs.cpu().numpy()
            batch_labels = np.argmax(outputs, axis=1)
            # 将ids和batch_labels组合成字典
            for i in range(len(ids)):
                id = ids[i].split(".")[0]
                preds[id] = batch_labels[i]

    df = pd.DataFrame()
    df["id"] = preds.keys()
    df["label"] = preds.values()
    df.to_csv("vgg16_test.csv", index=False)
