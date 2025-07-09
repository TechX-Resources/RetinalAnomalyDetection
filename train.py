import os
import time
import copy
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# utils.py for visuals
from utils import load_data, category_percentage, correlation_between_labels, venn_diagram

"""
- Transfer Learning: ResNet-50 pretrained on ImageNet is fine-tuned 
  for retinal disease classification.

- Layer Freezing: Only the final ResNet block (`layer4`) and the 
  fully connected layer (`fc`) are trained; all earlier layers are frozen. # so only updating/changing the last year for effiency

- Data Augmentation: Aggressive augmentations (flips, jitter, rotations) 
  are applied during training to increase robustness.
"""


class RetinalDisorderDataset(Dataset):
    def __init__(self, data_file, img_dir, transform=None):
        self.img_data = data_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_data.iloc[idx]['filename'])
        image = read_image(img_path)
        image = self.transform(image)
        # Ensure label values are float type (fix conversion error)
        label = self.img_data.iloc[idx, 1:].astype(float).values
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


def plot_loss_history(train_loss, val_loss):
    plt.figure(figsize=(20, 8))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, lw=3, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, lw=3, color='green', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()


def get_pos_weight(df):
    pos_weight = []
    for c in range(df.shape[1]):
        denom = (df.iloc[:, c] == 1).sum()
        weight = (df.iloc[:, c] == 0).sum() / denom if denom != 0 else 1.0
        pos_weight.append(weight)
    return pos_weight


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25, model_name=None):
    model_name = model_name if model_name else model.__class__.__name__

    if not os.path.exists('models'):
        os.mkdir('models')

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    loss_history = {'train': [], 'val': []}

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = torch.sigmoid(outputs)
                    preds_rounded = torch.round(preds)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds_rounded == labels).all(dim=1)).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            loss_history[phase].append(epoch_loss)

            if phase == 'train':
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                scheduler.step()
            else:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'models/{model_name}_best.pth')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model, loss_history


if __name__ == '__main__':
    # Load CSV data - adjust path as necessary
    train_df = load_data('data/train/train.csv', ',')

    # Labels excluding filename column
    disease_labels = train_df.columns[1:]

    # Data splits
    train_data, val_data = train_test_split(train_df, train_size=0.9, random_state=42)

    # Image transformations for augmentation and normalization
    img_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.RandomRotation(degrees=360),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    data_df = {'train': train_data, 'val': val_data}

    image_datasets = {x: RetinalDisorderDataset(data_file=data_df[x],
                                               img_dir='data/train/train/',
                                               transform=img_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=48,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = models.resnet50(pretrained=True)

    # Freeze layers except last block & fc
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(disease_labels))
    model = model.to(device)

    pos_weight = get_pos_weight(train_df.iloc[:, 1:])
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model, loss_history = train_model(model, criterion, optimizer, scheduler, dataloaders, device,
                                      num_epochs=30, model_name='ResNet50_Retinal')

    plot_loss_history(loss_history['train'], loss_history['val'])
