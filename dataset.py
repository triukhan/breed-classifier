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

        xmin, ymin, xmax, ymax = get_bbox(row['annotation_path'])
        img = img[ymin:ymax, xmin:xmax]

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['label'] - 1, dtype=torch.long)

        return img, label



########################
load_dotenv()
DATASET = os.getenv('DATASET_PATH')
mat_data = loadmat(f'{DATASET}/file_list.mat')

data = []
for num, img_path in enumerate(mat_data['file_list']):
    data.append({
        'img_path': f'{DATASET}/images/{str(img_path[0][0])}',
        'annotation_path': f'{DATASET}/annotation/{mat_data['annotation_list'][num][0][0]}',
        'label': mat_data['labels'][num][0]
    })

df = pd.DataFrame(data)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = BreedDataset(train_df, transform=train_transform)
val_dataset   = BreedDataset(val_df,   transform=val_transform)
test_dataset  = BreedDataset(test_df,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
