import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from PIL import Image

class Folder_Dataset(data.Dataset):
    def __init__(self, dir_data, transform=None):
        dir_root = os.getcwd()
        self.dir_root = dir_root
        self.dir_data = os.path.join(self.dir_root, dir_data)
        self.transform = transform
        _dataset = ImageFolder(self.dir_data, transform=self.transform)
        self.images = _dataset.samples
        self.labels = np.array(_dataset.targets)

    def __getitem__(self, index: int):
        img_path, label = self.images[index][0], self.labels[index]
        img_path = os.path.join(self.dir_root, img_path)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.tensor(label, dtype=torch.int64)
        return img, label

    def __len__(self):
        return len(self.images)
