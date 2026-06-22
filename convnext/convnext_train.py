import os
import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from scipy.io import loadmat
from torchvision import transforms
import torch
import torch.nn as nn
from dataset import BreedDataset
import timm

load_dotenv()
DATASET = os.getenv('DATASET_PATH')
mat_data = loadmat(f'{DATASET}/file_list.mat')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
data = []

for num, img_path in enumerate(mat_data['file_list']):
    data.append({
        'img_path': f'{DATASET}/images/{str(img_path[0][0])}',
        'annotation_path': f'{DATASET}/annotation/{mat_data['annotation_list'][num][0][0]}',
        'label': mat_data['labels'][num][0]
    })

df = pd.DataFrame(data)

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(384, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),

    transforms.RandAugment(num_ops=2, magnitude=9),

    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.225],
        [0.229, 0.224, 0.225]
    ),
])

train_dataset = BreedDataset(train_df, transform=train_transform)
val_dataset   = BreedDataset(val_df,   transform=val_transform)
test_dataset  = BreedDataset(test_df,  transform=val_transform)

g = torch.Generator()
g.manual_seed(42)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True, generator=g)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True, generator=g)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=4, pin_memory=True, generator=g)


def get_model(num_classes=120):
    mdl = timm.create_model(
        'convnextv2_base', pretrained=True, num_classes=num_classes
    )
    return mdl.cuda()


def train_epoch(mdl, loader, criterion, optimizer, ema):
    mdl.train()
    total_loss, correct = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        outputs = mdl(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        ema.update(mdl)

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def eval_epoch(mdl, loader, criterion):
    mdl.eval()
    total_loss, correct = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to('cuda'), labels.to('cuda')

        outputs = mdl(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def train(
    mdl,
    t_loader,
    v_loader,
    epochs,
    save_path='best_model.pth'
):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ema = ModelEmaV2(mdl, decay=0.999)
    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(mdl, t_loader, criterion, optimizer, ema)
        val_loss, val_acc = eval_epoch(ema.module, v_loader, criterion)

        scheduler.step()

        print(
            f'Epoch {epoch + 1}/{epochs} '
            f'| train loss: {train_loss:.4f} '
            f'acc: {train_acc:.4f} '
            f'| val loss: {val_loss:.4f} '
            f'acc: {val_acc:.4f}'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                ema.module.state_dict(),
                save_path
            )

            print(
                f'  -> saved best model '
                f'(val_acc={val_acc:.4f})'
            )


model = get_model(num_classes=120)
train(model, train_loader, val_loader, epochs=10, save_path='stage1_best.pth')

