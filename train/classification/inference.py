import os
import cv2
from pycocotools.coco import COCO
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class TestDataset(Dataset):
    def __init__(self, csv_path, img_paths, transform):
        self.transform = transform
        test_image = pd.read_csv(csv_path)
        self.image_files = test_image["image_id"]
        self.img_paths = img_paths

    def __getitem__(self, idx):
        image_path = self.img_paths + '/' + self.image_files[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transform is not None:
            img_transform = self.transform(image=img)
            img_transform = img_transform['image']
        else:
            img_transform = img

        return img_transform

    def __len__(self):
        return len(self.image_files)

def test(model_path, seed, device, img_size, test_dir):
    cate_json = '/opt/ml/project/final-project-level3-cv-12/dataset/instances_Validation.json'
    test_coco = COCO(cate_json)
    categories = test_coco.dataset['categories']

    # seed
    seed_everything(seed)

    image_csv = test_dir + '/crop.csv'
    
    # dataset, dataloader
    h, w = img_size
    test_dataset = TestDataset(image_csv, test_dir, transform=A.Compose([
        A.Resize(h, w), 
        A.Normalize(),
        ToTensorV2(),
        ]))
    test_loader = DataLoader(test_dataset,shuffle=False)

    # model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 158)

    ## model load
    checkpoint =torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    submission = pd.read_csv(image_csv)
    all_predictions = []
    label_names = []

    for images in test_loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            name = categories[pred]['name']
            all_predictions.extend(pred.cpu().numpy())
            label_names.append(name)
    submission['ans'] = all_predictions
    submission['name'] = label_names

    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--model_path', type=str, default = '/opt/ml/test/all_test/epoch0019_0203.pth', help='train config file path')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--test_dir', type=str, default = '/opt/ml/test/yolo_crop/KakaoTalk_20220519_165405194_01')

    parser.add_argument('--img_size', type=tuple, default=(384, 384), help='image size (height, width)')

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')


    args = parser.parse_args()

    return args

def main(args):
    test(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
