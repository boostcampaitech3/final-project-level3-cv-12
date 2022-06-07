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

from pycocotools.coco import COCO
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
    def __init__(self, json_path, folder_path, transform=None):
        self.coco = COCO(json_path)
        self.annotation_id = self.coco.getAnnIds()
        self.image_id = self.coco.getImgIds()
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.folder_path = folder_path

        self.transform = transform

    def __len__(self):
        return len(self.annotation_id)

    def __getitem__(self, idx):
        annotation_info = self.coco.loadAnns(idx+1)[0]
        cate_id = annotation_info['category_id']

        file_path = self.coco.loadImgs(annotation_info['image_id'])[0]['file_name']
        image_path = os.path.join(self.folder_path, file_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            x, y, w, h = annotation_info['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            crop_image = image[y:y+h, x:x+w]

            label = torch.tensor(cate_id-1)
            if self.transform is not None:
                img_transform = self.transform(image=crop_image)
                img_transform = img_transform['image']
            else:
                img_transform = crop_image
            return img_transform, label
        except Exception:
            return None


def save_model(model, saved_dir, file_name='fcn_resnet18_best_model(pretrained).pth'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model, output_path)

def collate_fn(batch):
    images, labels = [], []
    for data in batch:
        if data is not None:
            images.append(data[0])
            labels.append(data[1])
        else:
            pass
    padded_inputs = torch.nn.utils.rnn.pad_sequence(images, batch_first=True)
    return [padded_inputs.contiguous(), torch.stack(labels).contiguous()]


def train(seed, device, train_path, val_path, folder_path, epoch, img_size, saved_dir):
    # seed
    seed_everything(seed)

    # dataset, dataloader
    h, w = img_size
    t_dataset = CustomDataset(train_path, folder_path, transform=A.Compose([
        A.Resize(h, w), 
        A.Normalize(),
        ToTensorV2(),
        ]))

    v_dataset = CustomDataset(val_path, folder_path, transform=A.Compose([
        A.Resize(h, w), 
        A.Normalize(),
        ToTensorV2(),
        ]))

    train_dataloader = DataLoader(t_dataset, batch_size=64, shuffle=True, num_workers=1, collate_fn = collate_fn)
    val_dataloader = DataLoader(v_dataset, batch_size=64, shuffle=False, num_workers=1, collate_fn = collate_fn)

    # model
    model = models.resnet18(pretrained=True)
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
