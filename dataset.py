import os
import cv2
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import torch
from torchvision import transforms


def get_bbox(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bndbox = root.find('.//bndbox')
    return (
        int(bndbox.find('xmin').text),
        int(bndbox.find('ymin').text),
        int(bndbox.find('xmax').text),
        int(bndbox.find('ymax').text)
    )


class BreedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(row['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        xmin, ymin, xmax, ymax = get_bbox(row['annotation_path'])

        pad = 10

        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(w, xmax + pad)
        ymax = min(h, ymax + pad)

        img = img[ymin:ymax, xmin:xmax]

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['label'] - 1, dtype=torch.long)

        return img, label
