import os
import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import vit_base_patch16, vit_large_patch16, vit_huge_patch14

data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

class FoodDataSet(Dataset):
    def __init__(self, path, tfm=data_transform, files=None):
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
        id = fname.split("/")[-1].split(".")[0]
        im = Image.open(fname)
        im = self.transform(im)
        # im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # test has no label
        return im, label, id

def parse_args():
    """:arguments"""
    config = {
        # model config
        "gpu_id": 1,
        "model_name": "vit_base_patch16", # vit_base_patch16, vit_large_patch16, vit_huge_patch14
        "num_classes": 11,

        # testing config
        "batch_size": 1,
    }
    return config

def main(
    gpu_id,
    model_name,
    num_classes,
    batch_size,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_pth = "./{}.pth".format(model_name)
    assert os.path.exists(save_pth), "model path does not exist."

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../datasets"))
    image_path = os.path.join(data_root, "food11")
    test_path = os.path.join(image_path, "test")
    assert os.path.exists(test_path), "dataset path does not exist."

    # Load images
    test_dataset = FoodDataSet(test_path, tfm=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # create model
    if model_name == "vit_base_patch16":
        model = vit_base_patch16(num_classes=num_classes)
    elif model_name == "vit_large_patch16":
        model = vit_large_patch16(num_classes=num_classes)
    elif model_name == "vit_huge_patch14":
        model = vit_huge_patch14(num_classes=num_classes)
    else:
        raise ValueError("model name must be mobilenetv3_small or mobilenetv3_large. Got {}".format(model_name))
    model.to(device)
    model.load_state_dict(torch.load(save_pth, map_location=device))

    # Set the model to evaluation mode
    preds = {}
    model.eval()
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
    df['id'] = preds.keys()
    df['label'] = preds.values()
    df.to_csv("predictions.csv", index=False)



if __name__ == '__main__':
    main(**parse_args())