# -*- coding: utf-8 -*-


# ***************************************************
# * File        : files_dataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-26
# * Version     : 0.1.032622
# * Description : create img_dir dataset
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class CustomImageDataset(Dataset):

    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 transform = None, 
                 target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        return the number of samples in dataset
        """
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """
        loads and returns a sample from 
        the dataset at the given index `idx`
        """
        # # image tensor
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # image path
        image = read_image(img_path)
        # image label
        label = self.img_labels.iloc[idx, 1]
        # image transform
        if self.transform:
            image = self.transform(image)
        # image label transform
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


# dataset
train_dataset = CustomImageDataset(
    annotations_file = "",
    img_dir = "",
    transform = transforms.ToTensor(),
    target_transform = transforms.ToTensor(),
)
test_dataset = CustomImageDataset(
    annotations_file = "",
    img_dir = "",
    transform = transforms.ToTensor(),
    target_transform = transforms.ToTensor(),   
)

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 64, 
    shuffle = True,
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size = 64, 
    shuffle = False,
)

# ------------------------------
# test
# ------------------------------
# test data
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# test plot
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
