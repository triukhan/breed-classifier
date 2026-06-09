import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy.io import loadmat
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models

from dataset import BreedDataset

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
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

train_dataset = BreedDataset(train_df, transform=train_transform)
val_dataset   = BreedDataset(val_df,   transform=val_transform)
test_dataset  = BreedDataset(test_df,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


def get_model(num_classes=120):
    mdl = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in mdl.parameters():
        param.requires_grad = False

    for param in mdl.layer4.parameters():
        param.requires_grad = True

    for param in mdl.fc.parameters():
        param.requires_grad = True

    mdl.fc = nn.Linear(mdl.fc.in_features, num_classes) # because 120 classes

    return mdl.to('cuda')


def train_epoch(mdl, loader, criterion, optimizer):
    mdl.train()
    total_loss, correct = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = mdl(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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


def train(mdl, t_loader, v_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(mdl, t_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(mdl, v_loader, criterion)
        scheduler.step()

        print(f'Epoch {epoch+1}/{epochs} '
              f'| train loss: {train_loss:.4f} acc: {train_acc:.4f} '
              f'| val loss: {val_loss:.4f} acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(mdl.state_dict(), 'best_model.pth')
            print(f'  -> saved best model (val_acc={val_acc:.4f})')


model = get_model(num_classes=120)
train(model, train_loader, val_loader, epochs=10, lr=1e-4)