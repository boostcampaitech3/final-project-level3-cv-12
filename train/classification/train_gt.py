import os
import cv2
from torchvision import models

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from argparse import ArgumentParser


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class CustomDataset(Dataset):
    def __init__(self, csv_path, folder_path, mode = 'train', transform=None):
        data = pd.read_csv(csv_path)
        self.image_files = data["image_id"]
        self.image_labels = data["classes"]
        self.transform = transform
        self.crop_folder = os.path.join(folder_path, mode)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.crop_folder, self.image_files[idx])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        label = torch.tensor(self.image_labels[idx])
        if self.transform is not None:
            img_transform = self.transform(image=img)
            img_transform = img_transform['image']
        else:
            img_transform = img
        return img_transform, label

def save_model(model, saved_dir, file_name='fcn_resnet18_best_model(pretrained).pth'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model, output_path)

def train(seed, device, train_path, val_path, folder_path, epoch, img_size, saved_dir):
    # seed
    seed_everything(seed)
    
    # dataset, dataloader
    h, w = img_size
    t_dataset = CustomDataset(train_path, folder_path, mode='train', transform=A.Compose([
        A.Resize(h, w), 
        A.Blur(p=0.3),
        A.Rotate(10, p=0.2),
        A.Normalize(),
        ToTensorV2(),
        ]))

    v_dataset = CustomDataset(val_path, folder_path, mode='val', transform=A.Compose([
        A.Resize(384,384), 
        A.Normalize(),
        ToTensorV2(),
        ]))

    train_dataloader = DataLoader(t_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(v_dataset, batch_size=64, shuffle=False, num_workers=1)

    # model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 158)

    # optimizer & loss
    LEARNING_RATE = 0.0001
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) 

    # train & validation
    loader_type = {
        "train": train_dataloader,
        "test": val_dataloader
    }

    submission = pd.read_csv('/opt/ml/project/final-project-level3-cv-12/dataset/gt/val/crop.csv')

    model = model.to(device)
    best_test_acc = 0
    best_test_loss = 9999.

    for epoch in range(epoch):
        all_predictions = []

        for mode in ["train", "test"]:
            running_loss = 0
            running_acc = 0
            
            if mode == 'train':
                model.train()
            elif mode == 'test':
                model.eval()
            
            for idx, (images, labels) in enumerate(loader_type[mode]):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(mode == 'train'):
                    
                    logits = model(images)
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits, 1)
                    pred = logits.argmax(dim=-1)
                    if mode == 'test':
                        all_predictions.extend(pred.cpu().numpy())
                    
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                running_acc += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(loader_type[mode].dataset)
            epoch_acc = running_acc / len(loader_type[mode].dataset)
            
            print(f'epoch: {epoch+1}, {mode}-데이터 셋 Loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}')
            
            if mode == 'test' and best_test_acc < epoch_acc:
                best_test_acc = epoch_acc

            if mode == 'test' and best_test_loss > epoch_loss:
                file_name = f'epoch{epoch+1:04d}_{str(epoch_loss)[:5].replace(".","")}'
                save_model(model, saved_dir, f'{file_name}.pth')
                best_test_loss = epoch_loss
                submission['ans'] = all_predictions
        

    submission.to_csv(os.path.join(saved_dir, f'{file_name}.csv'), index=False)
    print("training end!!")
    print(f"best acc: {best_test_acc}, best loss: {best_test_loss}")


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--train_path', type=str, default = '/opt/ml/project/final-project-level3-cv-12/dataset/gt/train/crop.csv')
    parser.add_argument('--val_path', type=str, default = '/opt/ml/project/final-project-level3-cv-12/dataset/gt/val/crop.csv')
    parser.add_argument('--folder_path', type=str, default = '/opt/ml/project/final-project-level3-cv-12/dataset/gt')
    parser.add_argument('--saved_dir', type=str, default = '/opt/ml/project/final-project-level3-cv-12/train/classification/saved/resnet18')

    parser.add_argument('--img_size', type=tuple, default=(384, 384), help='image size (height, width)')
    parser.add_argument('--epoch', type=int, default=20, help='num epoch')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')


    args = parser.parse_args()

    return args

def main(args):
    train(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
