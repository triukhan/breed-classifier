import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from dataset import BreedDataset
from torchvision import models

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to('cuda'), labels.to('cuda')

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


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
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])
_, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
_, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
test_dataset  = BreedDataset(test_df,  transform=val_transform)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

mdl = models.resnet50(weights=None)
mdl.fc = nn.Linear(mdl.fc.in_features,120)
state_dict = torch.load('best_model.pth', map_location='cuda', weights_only=True)
mdl.load_state_dict(state_dict, strict=True)

print(eval_epoch(mdl.cuda(), test_loader, nn.CrossEntropyLoss()))
